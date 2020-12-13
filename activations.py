import torch.nn as nn

Activations = {
    "relu": nn.ReLU(),
#     "silu": silu,
#     "swish": silu,
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
#     "tanh": torch.tanh,
#     "gelu_new": gelu_new,
#     "gelu_fast": gelu_fast,
#     "mish": mish,
#     "linear": linear_act,
#     "sigmoid": torch.sigmoid,
}


