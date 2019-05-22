#!/usr/bin/env bash

cd ../src

epochs=60
nsamples=100
for nsamples in 10 100 1000 10000 -1; do
echo $(date -Iseconds -u) : started optimization with $nsamples examples
python train.py -d BavarianCrops -m DualOutputRNN --epsilon 10 --loss_mode early_reward --no-visdom -x nsamples --test_every_n_epochs $epochs --earliness_reward_power 1 --train_on trainvalid --test_on eval -r 64 -n 4 -e $epochs -s -1 -b 1024 --warmup-steps 100 --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl --dropout 0.5 -w 16 -i 0 -a .4 --store /tmp --nsamples $nsamples --overwrite
echo $(date -Iseconds -u) : ended optimization with $nsamples examples
done
