import torch
import torch.nn as nn

in_features: int = 2
out_features: int = 1
num_layers: int = 4
width: int = 50
epoch:int=10000
num_sample:int=100
num_plot:int=100
learning_rate:float = 1e-3
loss_weights:torch.Tensor =torch.tensor([1.])

Activation = nn.Tanh()