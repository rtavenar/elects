#!/usr/bin/env bash

for dataset in GunPoint Wafer;
do
python src/train.py -d $dataset -m Conv1D --epsilon 0 --loss_mode twophase_cross_entropy -x twophase_a06_startearly_xentr --earliness_reward_power 1 --train_on trainvalid --test_on test -e 60 -s 30 -b 128 --hyperparametercsv /data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv --dropout 0.5 -w 16 -i 0 -a .6 --store /data/ICML2019 --overwrite
python src/train.py -d $dataset -m Conv1D --epsilon 0 --loss_mode twophase_linear_loss -x twophase_a06_startearly_linear --earliness_reward_power 1 --train_on trainvalid --test_on test -e 60 -s 30 -b 128 --hyperparametercsv /data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv --dropout 0.5 -w 16 -i 0 -a .6 --store /data/ICML2019 --overwrite
done