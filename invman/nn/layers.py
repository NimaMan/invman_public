import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, activation=F.relu, num_hidden_layers=1,
                 output_activation=None):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()

        if type(hidden_dim) == int:
            self.num_layers = num_hidden_layers + 1
            self.layers_dims = num_hidden_layers * [hidden_dim]
        elif type(hidden_dim) == list:
            self.num_layers = len(hidden_dim) + 1
            self.layers_dims = hidden_dim
        else:
            raise NotImplementedError

        self.layers_dims.insert(0, input_dim)
        self.layers_dims.append(output_dim)
        assert self.num_layers == len(self.layers_dims) - 1

        for i in range(self.num_layers):
            self.layers.append(nn.Linear(self.layers_dims[i], self.layers_dims[i+1]))

    def forward(self, h):
        for layer in self.layers[:-1]:
            h = layer(h)
            h = self.activation(h)
        return self.layers[-1](h)


if __name__ == "__main__":

    mlp = MLP(input_dim=2, hidden_dim=[32, 16], output_dim=2)
    h = torch.randn(100, 2)
    h = mlp(h)