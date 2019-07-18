#!/usr/bin/env bash

cd ../src

epochs=200
logstore="/data/ec_runtimes"
for nsamples in 10 50 100 250 500 750 1000 2500 5000 7500 10000 -1; do
echo samples $nsamples > $logstore/nsamples$nsamples.log
echo $(date -Iseconds -u) : started optimization with $nsamples examples >> $logstore/nsamples$nsamples.log
python train.py -d BavarianCrops -m DualOutputRNN --epsilon 10 --loss_mode early_reward -x nsamples$nsamples --test_every_n_epochs $epochs --earliness_reward_power 1 --train_on trainvalid --test_on eval -r 64 -n 4 -e $epochs -s -1 -b 1024 --warmup-steps 50 --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl --dropout 0.5 -w 16 -i 0 -a .4 --store /data/ec_runtimes --nsamples $nsamples --overwrite
echo $(date -Iseconds -u) : ended optimization with $nsamples examples >> $logstore/nsamples$nsamples.log
done
