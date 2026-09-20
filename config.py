import torch
import torch.nn as nn

in_features: int = 1
out_features: int = 1
num_layers: int = 2
width: int = 100
epoch:int=10000
num_sample=1
num_plot:int=1000
loss_weights:torch.Tensor =torch.tensor([1.])

Activation = nn.Tanh()