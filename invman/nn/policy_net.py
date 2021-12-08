import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from invman.nn.es_module import ESModule
from invman.nn.utils import get_activation_function
from rga.utils import save_init_args


class PolicyNet(ESModule):
    @save_init_args
    def __init__(self, input_dim, hidden_dim, output_dim, activation="selu", output_activation=None):
        super(PolicyNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = get_activation_function(activation)
        self.output_activation = output_activation
        self.layers = nn.ModuleList()

        if type(hidden_dim) == int:
            self.num_layers = 1
            self.layers_dims = [hidden_dim]
        elif type(hidden_dim) == list:
            self.num_layers = len(hidden_dim)
            self.layers_dims = hidden_dim
        else:
            raise NotImplementedError

        self.layers.append(nn.Linear(in_features=input_dim, out_features=self.layers_dims[0]))
        for i in range(self.num_layers-1):
            self.layers.append(nn.Linear(self.layers_dims[i], self.layers_dims[i+1]))

        self.layers.append(nn.Linear(in_features=self.layers_dims[-1], out_features=output_dim))
        self.features = {}

    def forward(self, state, return_features=False):
        h = state
        for layer_idx, layer in enumerate(self.layers[:-1]):
            h = layer(h)
            h = self.activation(h)
            if return_features:
                self.features[layer_idx] = h.detach().numpy()
        logits = self.layers[-1](h)
        action = torch.argmax(logits)
        if return_features:
            self.features[layer_idx+1] = logits.detach().numpy()
            return action.item(), self.features

        return action.item()


if __name__ == "__main__":

    model = PolicyNet(input_dim=2, hidden_dim=[32, 16], output_dim=25, activation="selu")
    state = torch.FloatTensor([19, 8])
    action = model(state)
    print(action)



