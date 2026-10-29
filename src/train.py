import torch.nn as nn
import torch.optim as optim

from . import sampling
from . import loss
from . import config as cfg

def train_hard(model:nn.Module)->None:
    """
    ポアソンの方程式に関するPINNの学習用関数
    入力が2つ(x,y). 出力が一つ(φ)
    境界条件(外周で φ=0, 中心の4点で φ=1)はモデル側(PoissonFCNN)で固定している
    """
    optimizer=optim.Adam(model.parameters(),lr=cfg.LEARNING_RATE)
    # 格子点は毎回同じなので, ループの外で1回だけ作る
    X  = sampling.sample_grid_closed(cfg.divisions(),[1.0,1.0])
    X.requires_grad  = True

    for i in range(cfg.EPOCH):
        Y  = model(X)
        optimizer.zero_grad()
        l=loss.loss_hard(X,Y)
        l.backward()
        optimizer.step()
        if i % 100 == 0 or i == cfg.EPOCH-1:
            print(f"EPOCH:{i} loss:{l.item():.6e}")
 
            
def train_soft(model:nn.Module)->None:
    """
    ポアソンの方程式に関するPINNの学習用関数
    入力が2つ(x,y). 出力が一つ(φ)
    Y0,Y1は境界条件
    """
    optimizer=optim.Adam(model.parameters(),lr=cfg.LEARNING_RATE)
    for i in range(cfg.EPOCH):
        X  = sampling.sample_grid_closed(cfg.divisions(),[1.0,1.0])
        X0 = sampling.sample_square_boundry(cfg.divisions())
        X1 = sampling.center_points(cfg.divisions())
        
        X.requires_grad  = True
        X0.requires_grad = True
        X1.requires_grad = True
        
        Y  = model(X)
        Y0 = model(X0)
        Y1 = model(X1)
        
        optimizer.zero_grad()
        l=loss.loss_soft(X,Y,Y0,Y1)
        l.backward()
        optimizer.step()
        if i % 100 == 0 or i == cfg.EPOCH-1:
                    print(f"EPOCH:{i} loss:{l.item():.6e}")


def train_soft_center(model:nn.Module)->None:
    """
    ポアソンの方程式に関するPINNの学習用関数
    入力が2つ(x,y). 出力が一つ(φ)
    外周の φ=0 はモデル側(SoftCenterFCNN)で固定しているので, 外周の損失用の点はサンプルしない
    Y1は中心の4点の境界条件
    """
    optimizer=optim.Adam(model.parameters(),lr=cfg.LEARNING_RATE)
    # 格子点は毎回同じなので, ループの外で1回だけ作る
    X  = sampling.sample_grid_closed(cfg.divisions(),[1.0,1.0])
    X1 = sampling.center_points(cfg.divisions())
    X.requires_grad = True

    for i in range(cfg.EPOCH):
        Y  = model(X)
        Y1 = model(X1)

        optimizer.zero_grad()
        l=loss.loss_soft_center(X,Y,Y1)
        l.backward()
        optimizer.step()
        if i % 100 == 0 or i == cfg.EPOCH-1:
            print(f"EPOCH:{i} loss:{l.item():.6e}")