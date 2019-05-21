import ray
import ray.tune as tune
import argparse
import os
import sys
from datasets.UCR_Dataset import UCRDataset
from models.ConvShapeletModel import ConvShapeletModel
import torch
from utils.trainer import Trainer
from train import get_datasets_from_hyperparametercsv, getModel, getDataloader, readHyperparameterCSV
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

from argparse import Namespace



def main():
    # parse input arguments
    args = parse_args()
    run_experiment_on_datasets(args)

def parse_args():
    parser = argparse.ArgumentParser("e.g. execute: /data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv -b 16 -c 2 -g .25 --skip-processed -r /tmp")
    parser.add_argument(
        'hyperparametercsv', type=str, default="/data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv",
        help='csv containing hyper parameters')
    parser.add_argument(
        '-x', '--experiment', type=str, default="sota_comparison", help='Batch Size')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=96, help='Batch Size')
    parser.add_argument(
        '-c', '--cpu', type=int, default=2, help='number of CPUs allocated per trial run (default 2)')
    parser.add_argument(
        '-g', '--gpu', type=float, default=.2,
        help='number of GPUs allocated per trial run (can be float for multiple runs sharing one GPU, default 0.25)')
    parser.add_argument(
        '-r', '--local_dir', type=str, default=os.path.join(os.environ["HOME"],"ray_results"),
        help='ray local dir. defaults to $HOME/ray_results')
    parser.add_argument(
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    args, _ = parser.parse_known_args()
    return args

def run_experiment(args):
    """designed to to tune on the same datasets as used by Mori et al. 2017"""

    #experiment_name = args.dataset
    datasets = get_datasets_from_hyperparametercsv(args.hyperparametercsv)

    import random
    random.shuffle(datasets)

    if args.experiment == "early_reward":
        config = dict(
                batchsize=args.batchsize,
                workers=2,
                epochs=100, # will be overwritten by training_iteration criterion
                switch_epoch=-1,
                earliness_factor=tune.grid_search([0.75, 0.5, 0.25]),
                ptsepsilon=tune.grid_search([0, 5, 10]),
                hyperparametercsv=args.hyperparametercsv,
                warmup_steps=tune.grid_search([20, 10, 0]),
                dataset=tune.grid_search(datasets),
                drop_probability=tune.grid_search([0.75, 0.5, 0.25]),
                loss_mode="early_reward" # tune.grid_search(["twophase_linear_loss","twophase_cross_entropy"]),
            )

    if args.experiment == "twophase_cross_entropy":
        config = dict(
                batchsize=args.batchsize,
                workers=2,
                epochs=100, # will be overwritten by training_iteration criterion
                switch_epoch=50,
                earliness_factor=tune.grid_search([0.25, 0.5, 0.75]),
                ptsepsilon=tune.grid_search([0, 5, 10]),
                hyperparametercsv=args.hyperparametercsv,
                warmup_steps=tune.grid_search([5, 10, 20]),
                dataset=tune.grid_search(datasets),
                drop_probability=tune.grid_search([0.25, 0.5, 0.75]),
                loss_mode="twophase_cross_entropy" # tune.grid_search(["twophase_linear_loss","twophase_cross_entropy"]),
            )

    if args.experiment == "test":
        config = dict(
                batchsize=args.batchsize,
                workers=2,
                epochs=1, # will be overwritten by training_iteration criterion
                switch_epoch=-1,
                earliness_factor=tune.grid_search([0.5]),
                ptsepsilon=tune.grid_search([5]),
                hyperparametercsv=args.hyperparametercsv,
                warmup_steps=20,
                dataset="ECG200",
                drop_probability=tune.grid_search([0.8]),
                loss_mode="early_reward" # tune.grid_search(["twophase_linear_loss","twophase_cross_entropy"]),
            )

    if args.experiment == "sota_comparison":
        config = dict(
                batchsize=args.batchsize,
                workers=2,
                epochs=60, # will be overwritten by training_iteration criterion
                switch_epoch=30,
                earliness_factor=tune.grid_search([0.6, 0.7, 0.8, 0.9]),
                entropy_factor=tune.grid_search([0.01, 0.1, 0]),
                ptsepsilon=tune.grid_search([0, 10, 100]),
                hyperparametercsv=args.hyperparametercsv,
                dataset=tune.grid_search(datasets),
                drop_probability=0.5,
                loss_mode=tune.grid_search(["twophase_linear_loss","twophase_cross_entropy"]),
            )
    if args.experiment == "entropy_pts":
        config = dict(
                batchsize=args.batchsize,
                workers=2,
                epochs=60, # will be overwritten by training_iteration criterion
                switch_epoch=30,
                earliness_factor=tune.grid_search([0.6, 0.7, 0.8, 0.9]),
                entropy_factor=tune.grid_search([0, 0.01, 0.1]),
                ptsepsilon=tune.grid_search([0, 5, 10]),
                hyperparametercsv=args.hyperparametercsv,
                dataset=tune.grid_search(datasets),
                drop_probability=0.5,
                loss_mode="twophase_linear_loss",
            )

    if args.experiment == "phase1only":
        config = dict(
                batchsize=args.batchsize,
                workers=2,
                epochs=30, # will be overwritten by training_iteration criterion
                switch_epoch=999,
                earliness_factor=tune.grid_search([0.6, 0.7, 0.8, 0.9]),
                entropy_factor=0,
                ptsepsilon=0,
                hyperparametercsv=args.hyperparametercsv,
                dataset=tune.grid_search(datasets),
                drop_probability=tune.grid_search([0.2, 0.5, 0.75]),
                loss_mode="twophase_linear_loss",
            )

    tune.run_experiments(
        {
            args.experiment: {
                "resources_per_trial": {
                    "cpu": args.cpu,
                    "gpu": args.gpu,
                },
                'stop': {
                    'training_iteration': 1, # 1 iteration = 60 training epochs plus 1 eval epoch
                    'time_total_s':600 if not args.smoke_test else 1,
                },
                "run": RayTrainer,
                "num_samples": 1,
                "checkpoint_at_end": False,
                "config": config,
                "local_dir":args.local_dir
            }
        },
        verbose=0,
	resume=True)

def run_experiment_on_datasets(args):
    """
    Calls tune_dataset on each dataset listed in the datasetfile.

    :param args: argparse arguments forwarded further
    """
    datasets = get_datasets_from_hyperparametercsv(args.hyperparametercsv)
    resultsdir = os.path.join(args.local_dir)
    args.local_dir = resultsdir

    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    # start ray server
    #ray.init(redis_address="10.152.57.13:6379")
    if not ray.is_initialized():
        ray.init(include_webui=False, configure_logging=True, logging_level=logging.INFO)

    try:
        run_experiment(args)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print("error" + str(e))


class RayTrainer(ray.tune.Trainable):
    def _setup(self, config):
        self.dataset = config["dataset"]
        self.earliness_factor = config["earliness_factor"]

        hparams = pd.read_csv(config["hyperparametercsv"])

        # select only current dataset
        hparams = hparams.set_index("dataset").loc[config["dataset"]]

        config["learning_rate"] = float(hparams.learning_rate)
        config["num_layers"] = int(hparams.num_layers)
        config["hidden_dims"] = int(hparams.hidden_dims)
        config["shapelet_width_increment"] = int(hparams.shapelet_width_increment)

        logging.debug(hparams)
        logging.debug(config["batchsize"])

        self.epochs = config["epochs"]



        # handles multitxhreaded batching andconfig shuffling
        #self.traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=config["batchsize"], shuffle=True,
        #                                                   num_workers=config["workers"],
        #                                                   pin_memory=False)
        #self.validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=config["batchsize"], shuffle=False,
        #
        #                                              num_workers=config["workers"], pin_memory=False)

        # dict to namespace
        args = Namespace(**config)

        args.model = "Conv1D"
        args.shapelet_width_in_percent = False
        args.dropout = args.drop_probability

        self.traindataloader = getDataloader(dataset=args.dataset,
                                        partition="trainvalid",
                                        batch_size=config["batchsize"],
                                        num_workers=config["workers"],
                                        shuffle=True,
                                        pin_memory=True)


        self.validdataloader = getDataloader(dataset=args.dataset,
                                        partition="test",
                                        batch_size=config["batchsize"],
                                        num_workers=config["workers"],
                                        shuffle=False,
                                        pin_memory=True)


        args.nclasses = self.traindataloader.dataset.nclasses
        args.seqlength = self.traindataloader.dataset.sequencelength
        args.input_dims = self.traindataloader.dataset.ndims


        self.model = getModel(args)
        #self.model = ConvShapeletModel(num_layers=config["num_layers"],
        #                               hidden_dims=config["hidden_dims"],
        #                               ts_dim=1,
        #                               n_classes=nclasses,
        #                               use_time_as_feature=True,
        #                               drop_probability=config["drop_probability"],
        #                               scaleshapeletsize=False,
        #                               shapelet_width_increment=config["shapelet_width_increment"])

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # namespace to dict
        config = vars(args)
        config.pop("model") # delete string Conv1D to avoid confusion with model class

        self.config = config

        self.trainer = Trainer(self.model, self.traindataloader, self.validdataloader, **config)


    def _train(self):

        for epoch in range(self.epochs):
            self.trainer.new_epoch() # important for updating the learning rate
            stats = self.trainer.train_epoch(epoch)


        stats = self.trainer.test_epoch(dataloader=self.validdataloader)

        self.log(stats)

        return stats

    def log(self, stats):

        msg = "dataset {}, accuracy {:0.2f}, earliness {:0.2f}, mean_precision {:0.2f}, mean_recall {:0.2f}, kappa {:0.2f}, loss {:0.2f}"

        #print(self.dataset)
        #print(self.config)
        print(msg.format(self.dataset, stats["accuracy"], stats["earliness"], stats["mean_precision"], stats["mean_recall"],
                         stats["kappa"], stats["loss"]))


    def _save(self, path):
        path = path + ".pth"
        torch.save(self.model.state_dict(), path)
        return path

    def _restore(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)

if __name__=="__main__":

    main()
