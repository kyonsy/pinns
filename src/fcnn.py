import torch
import torch.nn as nn
import math as m
from . import config as cfg
from . import hard_constraint as hc


class FCNN(nn.Module):
    first  : nn.Linear
    hidden : nn.ModuleList
    final  : nn.Linear
    act    : nn.Module
    
    def __init__(self) -> None:
        super().__init__()
        self.first=nn.Linear(cfg.IN_FEATURES,cfg.WIDTH)
        self.hidden = nn.ModuleList(
            [nn.Linear(cfg.WIDTH, cfg.WIDTH) for _ in range(cfg.NUM_LAYERS)]
        )
        self.final = nn.Linear(cfg.WIDTH, cfg.OUT_FEATURES)
        self.act = cfg.ACTIVATION()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        linear: nn.Module
        h=self.first(X)
        for linear in self.hidden:
            h = linear(h)
            h = self.act(h)
        Y = self.final(h)
        return Y

"""
ポアソンの方程式用のFCNN
外周の境界で φ=0, 中心の周りの4点で φ=1 を厳密に満たすように出力を変換する
    φ = G + D*C*N
    D = 16x(1-x)y(1-y)          : 外周で0, 中心で1
    G = D/D(中心の4点)           : 外周で0, 中心の4点で1
    C = Π_k (1-exp(-|X-p_k|²/s²)) : 中心の4点 p_k で0, 離れると1
    N : ネットワーク本体の出力
"""
class PoissonFCNN(FCNN):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        N = super().forward(X)
        return hc.G(X) + hc.D(X) * hc.C(X) * N




if __name__=='__main__':
    from . import sampling
    
    model: FCNN = FCNN()
    X=sampling.sample_random_default() * 20
    X0=torch.zeros_like(X)
    
    X.requires_grad_(True)
    X0.requires_grad_(True)
    
    Y=model(X)
    Y0=model(X0)
    
    # optimizer.zero_grad()
    # l=loss.dumped_oscillation(X,Y,X0,Y0,0)