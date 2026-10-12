import torch
import torch.nn as nn
import math as m
from . import config as cfg


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
    
    P:torch.Tensor
    
    def __init__(self) -> None:
        super().__init__()
        n = cfg.GRID_DENSITY
        a = m.ceil(n/2)/n
        b = m.floor(n/2)/n
        self.register_buffer("P", torch.tensor([[a,a],[a,b],[b,a],[b,b]]))
        self.d_at_p = self.boundary_factor(self.P[:1]).item()
        self.s2 = (1/n)**2

    @staticmethod
    def boundary_factor(X: torch.Tensor) -> torch.Tensor:
        """外周で 0 になる関数 D"""
        x, y = X[:, 0:1], X[:, 1:2]
        return 16 * x * (1 - x) * y * (1 - y)
    
    def point_factor(self,X: torch.Tensor) -> torch.Tensor:
        """p_k で 0、離れると =1 になる関数 C"""
        r2 = ((X[:, None, :] - self.P[None, :, :])**2).sum(dim=2) # (N,4)
        return (1 - torch.exp(-r2 / self.s2)).prod(dim=1, keepdim=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        D = self.boundary_factor(X)
        G = D / self.d_at_p
        return G + D * self.point_factor(X) * super().forward(X)






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