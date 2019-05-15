import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy
import os
from models.EarlyClassificationModel import EarlyClassificationModel

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr and Marc Russwurm marc.russwurm[at]tum.de'

class ConvShapeletModel(EarlyClassificationModel):

    def __init__(self,
                 num_layers=1,
                 hidden_dims=50,
                 n_shapelets_per_size=None,
                 ts_dim=50,
                 n_classes=2,
                 load_from_disk=None,
                 use_time_as_feature=False,
                 drop_probability=0.5,
                 seqlength=100,
                 scaleshapeletsize=True,
                 shapelet_width_increment=10
                 ):

        super(ConvShapeletModel, self).__init__()
        self.d_model = hidden_dims*num_layers

        self.seqlength = seqlength
        self.scaleshapeletsize = scaleshapeletsize

        n_shapelets_per_size = build_n_shapelet_dict(num_layers, hidden_dims, shapelet_width_increment)
        self.nfeatures = sum(n_shapelets_per_size.values())

        self.shapelets = ShapeletBlocks(ts_dim, n_shapelets_per_size)

        #self.shapelets2 = ShapeletBlocks(self.nfeatures, n_shapelets_per_size)

        # dropout
        self.dropout_module = nn.Dropout(drop_probability)


        # batchnormalization after convolution
        #self.norm_module = nn.BatchNorm1d(self.nfeatures)
        self.norm_module = nn.BatchNorm1d(self.nfeatures)

        self.logreg_layer = nn.Linear(self.nfeatures, n_classes)
        self.decision_layer = nn.Linear(self.nfeatures, 1)

        torch.nn.init.normal_(self.decision_layer.bias, mean=-1e1, std=1e-1)

        self.n_shapelets_per_size = n_shapelets_per_size
        self.ts_dim = ts_dim
        self.n_classes = n_classes



    @property
    def n_shapelets(self):
        return sum(self.n_shapelets_per_size.values())

    def _logits(self, x):

        #shapelet_features = self._features(x)
        shapelet_features = self.shapelets.forward(x)
        #shapelet_features = self.shapelets2.forward(shapelet_features.transpose(1,2))

        shapelet_features = self.norm_module(shapelet_features.transpose(2, 1)).transpose(2, 1)
        shapelet_features = self.dropout_module(shapelet_features)

        logits = self.logreg_layer(shapelet_features)
        deltas = self.decision_layer(torch.sigmoid(shapelet_features))
        deltas = torch.sigmoid(deltas.squeeze(-1))
        pts, budget = self.attentionbudget(deltas)
        return logits, deltas, pts, budget

    def forward(self, x):
        logits, deltas, pts, budget = self._logits(x)
        logprobabilities = F.log_softmax(logits, dim=2)
        return logprobabilities, deltas, pts, budget

    def save(self, path="model.pth",**kwargs):
        print("Saving model to " + path)
        params = self.get_params()
        params["model_state"] = self.state_dict()
        #params["X_fit_"] = self.X_fit_
        #params["y_fit_"] = self.y_fit_
        # merge kwargs in params
        data = dict(
            params=params,
            config=kwargs
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data, path)

    def load(self, path):
        print("Loading model from " + path)
        data = torch.load(path, map_location="cpu")
        snapshot = data["params"]
        config = data["config"]
        model_state = snapshot.pop('model_state', snapshot)
        #self.X_fit_ = snapshot.pop('X_fit_', snapshot)
        #self.y_fit_ = snapshot.pop('y_fit_', snapshot)
        self.set_params(**snapshot)  # For hyper-parameters

        self._set_layers_and_optim()
        self.load_state_dict(model_state)
        #self.optimizer.load_state_dict(optimizer_state)
        self.eval()  # If at some point we wanted to use batchnorm or dropout

        for k,v in config.items():
            snapshot[k] = v

        return snapshot

def build_n_shapelet_dict(num_layers, hidden_dims, width_increments=10):
    """
    Builds a dictionary of format {<kernel_length_in_percentage_of_T>:<num_hidden_dimensions> , ...}
    returns n shapelets per size
    e.g., {10: 100, 20: 100, 30: 100, 40: 100}
    """
    n_shapelets_per_size = dict()
    for layer in range(num_layers):
        shapelet_width = (layer + 1) * width_increments  # in 10 feature increments of sequencelength percantage: 10 20 30 etc.
        n_shapelets_per_size[shapelet_width] = hidden_dims
    return n_shapelets_per_size

class ShapeletBlocks(nn.Module):

    def __init__(self, input_dim, n_shapelets_per_size):
        super(ShapeletBlocks, self).__init__()

        self.n_shapelets_per_size = n_shapelets_per_size

        blocks = list()
        for shapelet_size, n_shapelets_per_size in self.n_shapelets_per_size.items():
            conv = ShapeletConvolution(in_channels=input_dim,shapelet_size=shapelet_size, n_shapelets_per_size=n_shapelets_per_size)
            blocks.append(conv)

        self.shapelet_blocks = nn.ModuleList(blocks)

    def _temporal_pooling(self, x):
        pool_size = x.size(-1)
        pooled_x = nn.MaxPool1d(kernel_size=pool_size)(x)
        return pooled_x.view(pooled_x.size(0), -1)

    def get_shapelets(self):
        shapelets = []
        for block in self.shapelet_blocks:
            weights = block.weight.data.numpy()
            shapelets.append(numpy.transpose(weights, (0, 2, 1)))
        return shapelets

    def set_shapelets(self, l_shapelets):

        for shp, block in zip(l_shapelets, self.shapelet_blocks):
            block.weight.data = shp.view(block.weight.shape)

    def forward(self, x):
        sequencelength = x.shape[2]

        features_maxpooled = []
        for shp_sz, block in zip(self.n_shapelets_per_size.keys(), self.shapelet_blocks):
            f = block.forward(x)
            f_maxpooled = list()
            # sequencelength is not equal f.shape[2] -> f is based on padded input
            # -> padding influences length -> we take :sequencelength to avoid using inputs from the future at time t
            for t in range(1, sequencelength + 1):  # sequencelen
                f_maxpooled.append(self._temporal_pooling(f[:, :, :t]))
            f_maxpooled = torch.stack(f_maxpooled, dim=1)
            features_maxpooled.append(f_maxpooled)
        return torch.cat(features_maxpooled, dim=-1)

class ShapeletConvolution(nn.Module):
    """
    performs left side padding on the input and a convolution
    """
    def __init__(self, shapelet_size, in_channels, n_shapelets_per_size):
        super(ShapeletConvolution, self).__init__()

        # pure left padding to align classification time t with right edge of convolutional kernel
        self.pad = nn.ConstantPad1d((shapelet_size, 0), 0)
        self.conv = nn.Conv1d(in_channels=in_channels,
                  out_channels=n_shapelets_per_size,
                  kernel_size=shapelet_size)

    def forward(self, x):
        padded = self.pad(x)
        return self.conv(padded)
