
import torch.nn.functional as F
from .es_module import ESModule


def get_activation_function(activation="gelu"):
    if activation == "selu":
        return F.selu
    elif activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    else:
        raise NotImplementedError


def load_model(directory, device="cpu", strict=True):
    return ESModule.load(directory, device=device, strict=strict)
