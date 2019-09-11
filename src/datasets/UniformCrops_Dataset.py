import torch
import torch.utils.data
import os
import numpy as np

class UniformDataset(torch.utils.data.Dataset):

    def __init__(self, root, partition="train"):
        os.listdir(root)
        X = np.load(os.path.join(root,"X.npy"))
        y = np.load(os.path.join(root,"y.npy"))
        tt = np.load(os.path.join(root, "tt.npy")).astype(bool) # 0:train 1:test

        if partition in ["train","trainvalid","valid"]:
            self.X = X[~tt]
            self.y = y[~tt]
        elif partition in ["eval"]:
            self.X = X[tt]
            self.y = y[tt]

        _, self.sequencelength, self.ndims = self.X.shape
        self.nclasses = len(np.unique(y))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        X = self.X[idx]
        y = np.array(self.y[idx].repeat(self.sequencelength))

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        return X, y
