import torch
import torch.nn as nn
import torch.optim as optim

from . import sampling
from . import loss
from . import config as cfg

def train_dump_sys(model:nn.Module,t_range:float) -> None:
    optimizer=optim.Adam(model.parameters(),lr=cfg.learning_rate)
    for _ in range(cfg.epoch):
        X=sampling.sample_grid(cfg.grid_density,[t_range])
        X0=torch.zeros_like(X)
        
        X.requires_grad_(True)
        X0.requires_grad_(True)
        
        Y=model(X)
        Y0=model(X0)
        
        optimizer.zero_grad()
        l=loss.loss_dump_sys(X,Y,X0,Y0,0.0)
        l.backward()
        optimizer.step()
       
       
"""
ポアソンの方程式に関するPINNの学習用関数
入力が2つ(x,y). 出力が一つ(φ)
Y0,Y1は境界条件
""" 
def train_poisson_eq(model:nn.Module)->None:
    optimizer=optim.Adam(model.parameters(),lr=cfg.learning_rate)
    for i in range(cfg.epoch):
        X  = sampling.sample_grid(cfg.grid_density,[1.0,1.0])
        X0 = sampling.sample_square_boundry(cfg.num_sample)
        X1 = torch.tensor([[0.5,0.5]])
        
        X.requires_grad  = True
        X0.requires_grad = True
        X1.requires_grad = True
        
        Y  = model(X)
        Y0 = model(X0)
        Y1 = model(X1)
        
        optimizer.zero_grad()
        l=loss.loss_poisson_sys(X,Y,Y0,Y1)
        l.backward()
        optimizer.step()
        print(f"epoch:{i}")