import copy
import importlib
import json
import os
import shutil
import sys

import numpy as np
import torch
from torch.nn import Module


class ESModule(Module):

    def get_model_shapes(self):
        model_shapes = []
        for param in self.parameters():
            p = param.data.cpu().numpy()
            model_shapes.append(p.shape)
            param.requires_grad = False
        return model_shapes

    @property
    def model_shapes(self, ):
        return self.get_model_shapes()

    def set_model_params(self, flat_params):
        model_shapes = self.model_shapes
        idx = 0

        for i, param in enumerate(self.parameters()):
            delta = np.product(model_shapes[i])
            block = flat_params[idx: idx + delta]
            block = np.reshape(block, model_shapes[i])
            idx += delta
            block_data = torch.from_numpy(block).float()
            param.data = block_data

        return self

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print("loading file %s" % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def count_model_params(self):
        orig_model = copy.deepcopy(self)
        orig_params = []
        model_shapes = []
        for param in orig_model.parameters():
            p = param.data.cpu().numpy()
            model_shapes.append(p.shape)
            orig_params.append(p.flatten())
        orig_params_flat = np.concatenate(orig_params)

        return len(orig_params_flat)

    @property
    def num_params(self):
        return self.count_model_params()

    def get_model_flat_params(
        self,
    ):
        orig_model = copy.deepcopy(self)
        orig_params = []
        for param in orig_model.parameters():
            p = param.data.cpu().numpy()
            orig_params.append(p.flatten())
        orig_params_flat = np.concatenate(orig_params)
        return orig_params_flat

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
                m.reset_parameters()

        self.apply(weight_reset)