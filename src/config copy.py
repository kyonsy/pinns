import torch
import torch.nn as nn

in_features: int = 1
out_features: int = 1
num_layers: int = 10
width: int = 32
epoch:int=1000
num_sample:int=100
num_plot:int=1000
learning_rate:float = 1e-3
loss_weights:torch.Tensor =torch.tensor([1.])

Activation = nn.Tanh()