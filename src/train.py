import torch
from models.DualOutputRNN import DualOutputRNN
from models.ConvShapeletModel import ConvShapeletModel
from datasets.UCR_Dataset import UCRDataset
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
from datasets.Synthetic_Dataset import SyntheticDataset
import argparse
from argparse import Namespace
from utils.trainer import Trainer
from utils.optim import ScheduledOptim
import pandas as pd
import os
import numpy as np
from models.wavenet_model import WaveNetModel
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler, WeightedRandomSampler
from sampler.imbalanceddatasetsampler import ImbalancedDatasetSampler
from datasets.UniformCrops_Dataset import UniformDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d','--datasets', type=str,default=None, nargs='+', help='UCR_Datasets Datasets to train on. Multiple values are allowed. '
                                                'Will also define name of the experiment. '
                                                'if not specified, will use all datasets of hyperparametercsv'
                                                '(requires --hyperparametercsv)')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=32, help='Batch Size')
    parser.add_argument(
        '-m', '--model', type=str, default="DualOutputRNN", help='Model variant')
    parser.add_argument(
        '-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument(
        '-w', '--workers', type=int, default=4, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '-l', '--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument(
        '--train_on', type=str, default="train", help="dataset partition to train. Choose from 'train', 'valid', "
                                                      "'trainvalid', 'eval' (default 'train')")
    parser.add_argument(
        '--test_on', type=str, default="valid",
        help="dataset partition to train. Choose from 'train', 'valid', 'trainvalid', 'eval' (default 'valid')")
    parser.add_argument(
        '--classmapping', type=str, default=None,
        help="classmapping for the bavarian crops dataset line-wise text file of format (0,0,415)")
    parser.add_argument(
        '--dropout', type=float, default=.2, help='dropout probability of the rnn layer')
    parser.add_argument(
        '--epsilon', type=float, default=0, help='bias factor to add on P(t) as a regularization parameter')
    parser.add_argument(
        '-n', '--num_layers', type=int, default=1, help='number of stacked layers. will be interpreted as stacked '
                                                        'RNN layers for recurrent models and as number of convolutions'
                                                        'for convolutional models...')
    parser.add_argument(
        '-r', '--hidden_dims', type=int, default=32, help='number of hidden dimensions per layer stacked hidden dimensions')
    parser.add_argument(
        '--train-valid-split-seed', type=int, default=0,
        help='random seed for splitting of train and validation datasets. '
             'only applies for --train_on train and --test_on valid. see also --train-valid-ratio')
    parser.add_argument(
        '--train-valid-split-ratio', type=float, default=.75,
        help='ratio of splitting the train dataset in training and validation partitions. '
             'only applies for --train_on train and --test_on valid. see also --train-valid-split-seed')
    parser.add_argument(
        '--nsamples', type=int, default=None,
        help='use only n examples from the dataset (this artificially reduces the datasize)')
    parser.add_argument(
        '--augment_data_noise', type=float, default=0., help='augmentation data noise factor. defaults to 0.')
    parser.add_argument(
        '-a','--earliness_factor', type=float, default=1, help='earliness factor')
    parser.add_argument(
        '--entropy-factor', type=float, default=0, help='entropy factor')
    parser.add_argument(
        '--shapelet_width_increment', type=int, default=10,
        help='increments in shapelet width in either percent of total sequencelength '
             'by using --shapelet-width-in-percent flag. or in number of features.')
    parser.add_argument('--shapelet-width-in-percent', action='store_true', help="interpret shapelet_width as percentage of full sequence")
    parser.add_argument('--resume-optimizer', action='store_true',
                        help="resume optimizer state as well (may lead to smaller learning rates")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite automatic snapshots if they exist")
    parser.add_argument('--ndvi', action='store_true',
                        help="only applies do BavarianCrops dataset calculate and use only normalized difference vegetation index (NDVI)")
    parser.add_argument('--no-visdom', action='store_true',
                        help="dont send visdom logs")
    parser.add_argument('--skip', action='store_true',
                        help="skip dataset completely if already exists...")
    parser.add_argument(
        '-x', '--experiment', type=str, default="test", help='experiment prefix')
    parser.add_argument(
        '--hyperparametercsv', type=str, default=None, help='hyperparams csv file')
    parser.add_argument(
        '--store', type=str, default="/tmp", help='store run logger results')
    parser.add_argument(
        '--test_every_n_epochs', type=int, default=1, help='skip some test epochs for faster overall training')
    parser.add_argument(
        '--warmup-steps', type=int, default=None, help='warmupsteps for scheduled optimizer (if provided uses scheduled optimizer. otherwise regular adam)')
    parser.add_argument(
        '--earliness_reward_power', type=int, default=1, help='power of the y+ at the earliness term...')
    parser.add_argument(
        '--seed', type=int, default=None, help='seed for batching and weight initialization')
    parser.add_argument(
        '-i', '--show-n-samples', type=int, default=2, help='show n samples in visdom')
    parser.add_argument(
        '--loss_mode', type=str, default="twophase_early_simple", help='which loss function to choose. '
                                                                       'valid options are early_reward,  '
                                                                       'twophase_early_reward, '
                                                                       'twophase_linear_loss, or twophase_early_simple')
    parser.add_argument(
        '-s', '--switch_epoch', type=int, default=None, help='epoch at which to switch the loss function '
                                                             'from classification training to early training')

    args, _ = parser.parse_known_args()

    if args.switch_epoch is None:
        args.switch_epoch = args.epochs

    args = parse_dataset_names(args)

    return args

def parse_dataset_names(args):
    if args.hyperparametercsv is not None:
        datasets_with_hyperparameters = get_datasets_from_hyperparametercsv(args.hyperparametercsv)

    if args.datasets is not None and args.hyperparametercsv is not None:
        missing_datasets = [dataset for dataset in args.datasets if dataset not in datasets_with_hyperparameters]
        if len(missing_datasets) > 0:
            msg = "Datasets {} not present in hyperparametercsv {}. Please choose from {}"
            raise ValueError(msg.format(", ".join(missing_datasets),
                                        args.hyperparametercsv,
                                        ", ".join(datasets_with_hyperparameters)))

    if args.datasets is None and args.hyperparametercsv is not None:
        datasets_with_hyperparameters.reverse() # from A-Z
        args.datasets = datasets_with_hyperparameters

    if args.datasets is None and args.hyperparametercsv is None:
        raise ValueError("No --datasets and --hyperparametercsv provided. Either specifify the specific dataset "
                         "or provide a list of datasets in --hyperparametercsv")

    return args


def readHyperparameterCSV(args):
    print("reading "+args.hyperparametercsv)

    # get hyperparameters from the hyperparameter file for the current dataset...
    hparams = pd.read_csv(args.hyperparametercsv)

    # select only current dataset
    hparams = hparams.set_index("dataset").loc[args.dataset]

    args_dict = vars(args)

    for key,value in hparams.iteritems():

        # ignore empty columns
        if 'Unnamed' not in key:

            # only overwrite if key exists in parsed arguments
            if key in args_dict.keys():
                datatype_function = type(args_dict[key])
                value = datatype_function(value)

                args_dict[key] = value
                print("overwriting {key} with {value}".format(key=key,value=value))

    return Namespace(**args_dict)

def train(args):

    if args.hyperparametercsv is not None:
        args = readHyperparameterCSV(args)

    region = "HOLL_2018_MT_pilot"

    traindataloader = getDataloader(dataset=args.dataset,
                                    partition=args.train_on,
                                    batch_size=args.batchsize,
                                    num_workers=args.workers,
                                    shuffle=True,
                                    pin_memory=True,
                                    train_valid_split_ratio=args.train_valid_split_ratio,
                                    train_valid_split_seed=args.train_valid_split_seed,
                                    region=region,
                                    classmapping=args.classmapping,
                                    seed=args.seed,
                                    ndvi=args.ndvi,
                                    nsamples=args.nsamples)

    testdataloader = getDataloader(dataset=args.dataset,
                                   partition=args.test_on,
                                   batch_size=args.batchsize,
                                   num_workers=args.workers,
                                   shuffle=False,
                                   pin_memory=True,
                                   train_valid_split_ratio=args.train_valid_split_ratio,
                                   train_valid_split_seed=args.train_valid_split_seed,
                                   region=region,
                                   classmapping=args.classmapping,
                                   seed=args.seed,
                                   ndvi=args.ndvi,
                                   nsamples=args.nsamples)

    #evaldataloader = getDataloader(dataset=args.dataset,
    #                               partition="eval",
    #                               batch_size=args.batchsize,
    #                               num_workers=args.workers,
    #                               shuffle=False,
    #                               pin_memory=True,
    #                               train_valid_split_ratio=args.train_valid_split_ratio,
    #                               train_valid_split_seed=args.train_valid_split_seed,
    #                               region=region)

    args.nclasses = traindataloader.dataset.nclasses
    args.seqlength = traindataloader.dataset.sequencelength
    args.input_dims = traindataloader.dataset.ndims
    model = getModel(args)
    # np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    if not args.no_visdom:
        visdomenv = "{}_{}_{}".format(args.experiment, args.dataset, args.loss_mode.replace("_","-"))
    else:
        visdomenv = None

    print("Visdom Environment: {}".format(visdomenv))

    config = dict(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        earliness_factor=args.earliness_factor,
        visdomenv=visdomenv,
        switch_epoch=args.switch_epoch,
        loss_mode=args.loss_mode,
        show_n_samples=args.show_n_samples,
        store=os.path.join(args.store,args.experiment,args.dataset),
        overwrite=args.overwrite,
        ptsepsilon=args.epsilon,
        test_every_n_epochs=args.test_every_n_epochs,
        entropy_factor = args.entropy_factor,
        resume_optimizer = args.resume_optimizer,
        warmup_steps = args.warmup_steps,
        earliness_reward_power=args.earliness_reward_power
    )

    trainer = Trainer(model,traindataloader,testdataloader,**config)
    trainer.fit()


    #stats = trainer.test_epoch(evaldataloader)

    pass

def getDataloader(dataset, partition, train_valid_split_ratio=0.75,seed=None, ndvi=False, nsamples=None, **kwargs):

    if seed is None:
        # ensure the batchsing sequence is the same at different runs
        # useful for qualitative experiments to see identical samples in visdom...
        seed = sum([ord(ch) for ch in partition])

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if dataset == "synthetic":
        torchdataset = SyntheticDataset(num_samples=2000, T=100)
    elif dataset == "BavarianCrops":
        root = "/ssd/BavarianCrops"
        torchdataset = BavarianCropsDataset(root=root, region="holl", partition=partition, nsamples=nsamples, classmapping=kwargs["classmapping"], ndvi=ndvi)
    elif "UniformCrops" in dataset:

        nsamples = dataset.split("_")[-1]

        root = "/home/marc/projects/EarlyClassification/Databases/"
        dataset = "BavarianCrops_uniform_" + nsamples

        torchdataset = UniformDataset(os.path.join(root,dataset), partition=partition)
    else:
        torchdataset = UCRDataset(dataset, partition=partition, ratio=train_valid_split_ratio, randomstate=seed)

    if kwargs["shuffle"]:
        #sampler = WeightedRandomSampler(torchdataset.dataweights, len(torchdataset.dataweights),  replacement=True)
        #sampler = ImbalancedDatasetSampler(torchdataset)
        sampler = RandomSampler(torchdataset)
    else:
        sampler = SequentialSampler(torchdataset)

    # for training dataset -> shuffle is true

    dataloader = torch.utils.data.DataLoader(dataset=torchdataset,
                                             #batch_sampler=BatchSampler(sampler, kwargs["batch_size"], drop_last=False),
                                             sampler=sampler,
                                             batch_size=kwargs["batch_size"],
                                             num_workers=kwargs["num_workers"],
                                             pin_memory=kwargs["pin_memory"])

    return dataloader

def getModel(args):
    # Get Model
    if args.model == "DualOutputRNN":
        model = DualOutputRNN(input_dim=args.input_dims, nclasses=args.nclasses, hidden_dims=args.hidden_dims,
                              num_rnn_layers=args.num_layers, dropout=args.dropout, init_late=True)

    elif args.model == "WaveNet":

        model = WaveNetModel(
                 layers=5,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=args.nclasses,
                 classes=args.nclasses,
                 output_length=1,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 input_dims=args.input_dims,
                 bias=False)

    elif args.model == "Conv1D":
        model = ConvShapeletModel(num_layers=args.num_layers,
                                  hidden_dims=args.hidden_dims,
                                  ts_dim=args.input_dims,
                                  n_classes=args.nclasses,
                                  use_time_as_feature=False,
                                  seqlength=args.seqlength,
                                  scaleshapeletsize=args.shapelet_width_in_percent,
                                  drop_probability=args.dropout,
                                  shapelet_width_increment=args.shapelet_width_increment)
    else:
        raise ValueError("Invalid Model, Please insert either 'DualOutputRNN', or 'Conv1D'")

    if torch.cuda.is_available():
        model = model.cuda()

    return model

def get_datasets_from_hyperparametercsv(hyperparametercsv):
    return list(pd.read_csv(hyperparametercsv)["dataset"])

if __name__=="__main__":

    args = parse_args()
    for args.dataset in args.datasets:

        if os.path.exists(os.path.join(args.store,args.dataset)) and args.skip:
            print("path exists. skipping "+args.dataset)
            continue

        train(args)
