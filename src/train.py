import torch
import torch.nn as nn
import torch.optim as optim

from . import sampling
from . import loss
from . import config as cfg

def train_dump_sys(model:nn.Module,trange:float,zeta:float) -> None:
    optimizer=optim.Adam(model.parameters(),lr=cfg.learning_rate)
    # 格子点は毎回同じなので, ループの外で1回だけ作る
    X=sampling.sample_grid(cfg.grid_density,[trange])
    X0=torch.zeros_like(X)
    
    X.requires_grad_(True)
    X0.requires_grad_(True)
    
    for _ in range(cfg.epoch):
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
def train_poisson_eq(model:nn.Module)->None:
    optimizer=optim.Adam(model.parameters(),lr=cfg.learning_rate)
    # 格子点は毎回同じなので, ループの外で1回だけ作る
    X  = sampling.sample_grid(cfg.grid_density,[1.0,1.0])
    X.requires_grad  = True
    
    for i in range(cfg.epoch):
        Y  = model(X)
        
        optimizer.zero_grad()
        l=loss.loss_poisson_sys(X,Y)
        l.backward()
        optimizer.step()
        if i % 1000 == 0 or i == cfg.epoch-1:
            print(f"epoch:{i} loss:{l.item():.6e}")