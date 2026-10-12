import math as m

import torch
import torch.nn as nn
import torch.optim as optim

from . import sampling
from . import loss
from . import config as cfg

def train_dump_sys(model:nn.Module,trange:float,zeta:float) -> None:
    optimizer=optim.Adam(model.parameters(),lr=cfg.LEARNING_RATE)
    # 格子点は毎回同じなので, ループの外で1回だけ作る
    X=sampling.sample_grid(cfg.GRID_DENSITY,[trange])
    X0=torch.zeros_like(X)
    
    X.requires_grad_(True)
    X0.requires_grad_(True)
    
    for _ in range(cfg.EPOCH):
        Y=model(X)
        Y0=model(X0)
        
        optimizer.zero_grad()
        l=loss.loss_dump_sys(X,Y,X0,Y0,zeta)
        l.backward()
        optimizer.step()
       
       
"""
ポアソンの方程式に関するPINNの学習用関数
入力が2つ(x,y). 出力が一つ(φ)
境界条件(外周で φ=0, 中心の4点で φ=1)はモデル側(PoissonFCNN)で固定している
""" 
def train_poisson_hard(model:nn.Module)->None:
    optimizer=optim.Adam(model.parameters(),lr=cfg.LEARNING_RATE)
    # 格子点は毎回同じなので, ループの外で1回だけ作る
    X  = sampling.sample_grid(cfg.GRID_DENSITY,[1.0,1.0])
    X.requires_grad  = True

    for i in range(cfg.EPOCH):
        Y  = model(X)
        optimizer.zero_grad()
        l=loss.loss_poisson_hard(X,Y)
        l.backward()
        optimizer.step()
        if i % 1000 == 0 or i == cfg.EPOCH-1:
            print(f"EPOCH:{i} loss:{l.item():.6e}")
 
            
"""
ポアソンの方程式に関するPINNの学習用関数
入力が2つ(x,y). 出力が一つ(φ)
Y0,Y1は境界条件
""" 
def train_poisson_soft(model:nn.Module)->None:
    phi1 = m.ceil(cfg.GRID_DENSITY/2)/cfg.GRID_DENSITY
    phi2 = m.floor(cfg.GRID_DENSITY/2)/cfg.GRID_DENSITY
    
    optimizer=optim.Adam(model.parameters(),lr=cfg.LEARNING_RATE)
    for i in range(cfg.EPOCH):
        X  = sampling.sample_grid(cfg.GRID_DENSITY,[1.0,1.0])
        X0 = sampling.sample_square_boundry(cfg.GRID_DENSITY)
        X1 = torch.tensor([[phi1,phi1],[phi1,phi2],[phi2,phi1],[phi2,phi2]],device=cfg.DEVICE)
        
        X.requires_grad  = True
        X0.requires_grad = True
        X1.requires_grad = True
        
        Y  = model(X)
        Y0 = model(X0)
        Y1 = model(X1)
        
        optimizer.zero_grad()
        l=loss.loss_poisson_soft(X,Y,Y0,Y1)
        l.backward()
        optimizer.step()
        if i % 1000 == 0 or i == cfg.EPOCH-1:
                    print(f"EPOCH:{i} loss:{l.item():.6e}")