import torch
import torch.nn as nn
import torch.optim as optim
import math as m

from . import sampling
from . import loss
from . import config as cfg

def train_dump_sys(model:nn.Module,trange:float,zeta:float) -> None:
    optimizer=optim.Adam(model.parameters(),lr=cfg.learning_rate)
    for _ in range(cfg.epoch):
        X=sampling.sample_grid(cfg.grid_density,[trange])
        X0=torch.zeros_like(X)
        
        X.requires_grad_(True)
        X0.requires_grad_(True)
        
        Y=model(X)
        Y0=model(X0)
        
        optimizer.zero_grad()
        l=loss.loss_dump_sys(X,Y,X0,Y0,zeta)
        l.backward()
        optimizer.step()
       
       
"""
ポアソンの方程式に関するPINNの学習用関数
入力が2つ(x,y). 出力が一つ(φ)
Y0,Y1は境界条件
""" 
def train_poisson_eq(model:nn.Module)->None:
    phi1 = m.ceil(cfg.grid_density/2)/cfg.grid_density
    phi2 = m.floor(cfg.grid_density/2)/cfg.grid_density
    
    optimizer=optim.Adam(model.parameters(),lr=cfg.learning_rate)
    for i in range(cfg.epoch):
        X  = sampling.sample_grid(cfg.grid_density,[1.0,1.0])
        X0 = sampling.sample_square_boundry(cfg.grid_density)
        X1 = torch.tensor([[phi1,phi1],[phi1,phi2],[phi2,phi1],[phi2,phi2]])
        
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