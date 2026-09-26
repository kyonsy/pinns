import torch
import torch.nn as nn

in_features: int = 2
out_features: int = 1
num_layers: int = 8
width: int = 32
epoch:int=10000
num_sample:int=10
num_plot:int=1000
learning_rate:float = 1e-3
loss_weights:torch.Tensor =torch.tensor([1.])

Activation = nn.Tanh()