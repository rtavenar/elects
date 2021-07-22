End-to-end Learning for Early Classification of Time Series (ELECTS)
===

**Note: This repository was used for explorative development of models and loss functions. For a more distilled and easier runnable version of the main RNN model and BavarianCrops dataset, check [https://github.com/marccoru/elects](https://github.com/marccoru/elects)**

Execute single run
```angular2
-d BavarianCrops -m DualOutputRNN --epsilon 10 --loss_mode early_reward -x test --earliness_reward_power 1 --train_on train --test_on valid -r 64 -n 4 -e 60 -s -1 -b 1024 --warmup-steps 100 --classmapping /home/marc/data/BavarianCrops/classmapping.csv.holl --dropout 0.5 -w 16 -i 1 -a .4 --store /tmp/test --overwrite
```

<img width=200px src="docs/conv1d.png"/>

### Runs (visdom)

Gunpoint

<img width=100% src="docs/GunPoint_run.png"/>

Wafer

<img width=100% src="docs/Wafer_run.png"/>

EGC

<img width=100% src="docs/EGC_run.png"/>

Remote Sensing Dataset

<img width=100% src="docs/visdom_bavariancrops.png"/>



### Download data

```bash
wget https://s3.eu-central-1.amazonaws.com/corupublic/early_rnn.zip
```
