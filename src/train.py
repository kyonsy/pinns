from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from . import sampling
from . import loss
from .config import Config

def _fit(model:nn.Module,cfg:Config,compute_loss:Callable[[],torch.Tensor])->None:
    """
    学習ループ本体. compute_loss() でその回の損失を計算し, Adam で cfg.EPOCH 回更新する
    100回ごとと最後に損失を表示する
    """
    optimizer=optim.Adam(model.parameters(),lr=cfg.LEARNING_RATE)
    for i in range(cfg.EPOCH):
        optimizer.zero_grad()
        l=compute_loss()
        l.backward()
        optimizer.step()
        if i % 100 == 0 or i == cfg.EPOCH-1:
            print(f"EPOCH:{i} loss:{l.item():.6e}")


def train_hard(model:nn.Module,cfg:Config)->None:
    """
    ポアソンの方程式に関するPINNの学習用関数
    入力が2つ(x,y). 出力が一つ(φ)
    境界条件(外周で φ=0, 中心の4点で φ=1)はモデル側(PoissonFCNN)で固定している
    """
    X  = sampling.sample_grid_closed(cfg.divisions(),[1.0,1.0])
    X.requires_grad = True
    _fit(model,cfg,lambda: loss.loss_hard(X,model(X)))


def train_soft(model:nn.Module,cfg:Config)->None:
    """
    ポアソンの方程式に関するPINNの学習用関数
    入力が2つ(x,y). 出力が一つ(φ)
    外周 X0 と中心の4点 X1 の境界条件は損失関数で課す
    """
    X  = sampling.sample_grid_closed(cfg.divisions(),[1.0,1.0])
    X0 = sampling.sample_square_boundry(cfg.divisions())
    X1 = sampling.center_points(cfg.divisions())
    X.requires_grad = True
    _fit(model,cfg,lambda: loss.loss_soft(X,model(X),model(X0),model(X1),cfg.LOSS_WEIGHTS))


def train_soft_center(model:nn.Module,cfg:Config)->None:
    """
    ポアソンの方程式に関するPINNの学習用関数
    入力が2つ(x,y). 出力が一つ(φ)
    外周の φ=0 はモデル側(SoftCenterFCNN)で固定しているので, 外周の損失用の点はサンプルしない
    中心の4点 X1 の境界条件は損失関数で課す
    """
    X  = sampling.sample_grid_closed(cfg.divisions(),[1.0,1.0])
    X1 = sampling.center_points(cfg.divisions())
    X.requires_grad = True
    _fit(model,cfg,lambda: loss.loss_soft_center(X,model(X),model(X1),cfg.LOSS_WEIGHTS))
