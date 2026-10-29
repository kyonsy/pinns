import torch
import torch.nn as nn
from .config import Config
from . import hard_constraint as hc


class FCNN(nn.Module):
    first  : nn.Linear
    hidden : nn.ModuleList
    final  : nn.Linear
    act    : nn.Module
    n      : int        # 格子の分割数. ハード制約の関数に渡す
    
    def __init__(self, cfg:Config) -> None:
        super().__init__()
        self.first=nn.Linear(cfg.IN_FEATURES,cfg.WIDTH)
        self.hidden = nn.ModuleList(
            [nn.Linear(cfg.WIDTH, cfg.WIDTH) for _ in range(cfg.NUM_LAYERS)]
        )
        self.final = nn.Linear(cfg.WIDTH, cfg.OUT_FEATURES)
        self.act = cfg.ACTIVATION()
        self.n = cfg.divisions()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        linear: nn.Module
        h=self.first(X)
        for linear in self.hidden:
            h = linear(h)
            h = self.act(h)
        Y = self.final(h)
        return Y

class PoissonFCNN1(FCNN):
    """
    ポアソンの方程式用のFCNN
    外周の境界で φ=0, 中心の周りの4点で φ=1 を厳密に満たすように出力を変換する
        φ = G + D*C*N
        D = 16x(1-x)y(1-y)          : 外周で0, 中心で1
        G = D/D(中心の4点)           : 外周で0, 中心の4点で1
        C = Π_k (1-exp(-|X-p_k|²/s²)) : 中心の4点 p_k で0, 離れると1
        N : ネットワーク本体の出力
    """
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        N = super().forward(X)
        return hc.G(X,self.n) + hc.D(X) * hc.C(X,self.n) * N

class PoissonFCNN2(FCNN):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        N = super().forward(X)
        return hc.G(X,self.n) * hc.L(X,self.n) + hc.D(X) * hc.C(X,self.n) * N

class SoftCenterFCNN(FCNN):
    """
    ソフト制約用のFCNN (外周だけハード制約)
    外周の φ=0 は出力の変換で厳密に満たし, 中心の4点の φ=1 は損失関数で課す
        φ = D*N
        D = 16x(1-x)y(1-y) : 外周で0
        N : ネットワーク本体の出力
    """
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        N = super().forward(X)
        return hc.D(X) * N
