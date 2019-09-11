#!/bin/#!/usr/bin/env bash

epochs=200
logstore="/data/ec_runtimes_uniform"

for nsamples in 50 75 100 250 500 750 1000 2500 5000;
do

echo samples $nsamples > $logstore/UniformCrops_$nsamples.log
echo $(date -Iseconds -u) : started optimization with $nsamples examples >> $logstore/UniformCrops_$nsamples.log

python ../src/train.py -d UniformCrops_$nsamples -m DualOutputRNN --epsilon 10 --test_every_n_epochs $epochs --loss_mode early_reward -x "" -b 1024 --warmup-steps 100 --dropout 0.5 -w 16 -i 1 -a .25 --store $logstore --no-visdom --overwrite --train_on train --test_on eval -r 64 -n 4 -e $epochs -s -1

echo $(date -Iseconds -u) : ended optimization with $nsamples examples >> $logstore/UniformCrops_$nsamples.log

done