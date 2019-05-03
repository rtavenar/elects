import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


run = "/data/EV2019/early_reward"
states = os.path.join(run,"BavarianCrops","npy")
target = "/home/marc/projects/EV2019/images/example"

path = "/data/EV2019"

pass

file = "{array}_{epoch}.npy"
epoch = 100
sample = -1

def load_states(path, alpha, epsilon, nrrun):
    name = "{path}/earlyreward-alpha{alpha}-epsilon{epsilon}-run{run}"

    run = name.format(path=path, alpha=alpha, epsilon=epsilon, run=nrrun)

    data = pd.read_csv(os.path.join(run,"BavarianCrops","log_earliness.csv"))
    data = data.loc[data["mode"]=="test"]

    inputs = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="inputs",epoch=epoch)))
    probas = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="probas",epoch=epoch)))
    tstops = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="t_stops",epoch=epoch)))
    pts = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="pts",epoch=epoch)))
    deltas = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="deltas",epoch=epoch)))
    budgets = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="budget",epoch=epoch)))
    targets = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="targets",epoch=epoch)))
    predictions = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="predictions",epoch=epoch)))

    T = inputs.shape[1]

    input = inputs[sample]
    proba = probas[:,sample,:]
    tstop = tstops[sample] / T # normalize from idx 0-70 to float 0-1
    pt = pts[sample]
    delta = deltas[sample]
    budget = budgets[sample]
    label = targets[sample,0]
    prediction =predictions[sample]

load_states(path, alpha=0.4, epsilon=0, nrrun=1)
pass

