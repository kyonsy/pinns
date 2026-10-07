import torch
import torch.nn as nn
import math as m
from . import config as cfg


class FCNN(nn.Module):
    first: nn.Linear
    hidden: nn.ModuleList
    final: nn.Linear
    activations: nn.ModuleDict

    def __init__(self) -> None:
        super().__init__()
        self.first=nn.Linear(cfg.in_features,cfg.width)
        self.hidden = nn.ModuleList(
            [nn.Linear(cfg.width, cfg.width) for _ in range(cfg.num_layers)]
        )
        self.final = nn.Linear(cfg.width, cfg.out_features)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        linear: nn.Module
        Y=self.first(X)
        for linear in self.hidden:
            Y = linear(Y)
            Y = cfg.Activation(Y)
        Y = self.final(Y)
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
    def __init__(self) -> None:
        super().__init__()
        a = m.ceil(cfg.grid_density/2)/cfg.grid_density
        b = m.floor(cfg.grid_density/2)/cfg.grid_density
        self.register_buffer("P", torch.tensor([[a,a],[a,b],[b,a],[b,b]]))
        self.Dc = 16 * a * (1 - a) * b * (1 - b)
        self.s  = 1/cfg.grid_density

    @staticmethod
    def dist_boundary(X: torch.Tensor) -> torch.Tensor:
        x, y = X[:, 0:1], X[:, 1:2]
        return 16 * x * (1 - x) * y * (1 - y)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        D = self.dist_boundary(X)
        G = D / self.Dc
        r2 = ((X[:, None, :] - self.P[None, :, :])**2).sum(dim=2)
        C = (1 - torch.exp(-r2 / self.s**2)).prod(dim=1, keepdim=True)
        return G + D * C * super().forward(X)

if __name__=='__main__':
    from . import fcnn
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