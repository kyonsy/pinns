import torch
import torch.nn as nn
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
単位正方形の境界で φ=0 を厳密に満たすように, 出力に境界で0となる関数を掛ける
16x(1-x)y(1-y) は中心で1, 境界で0
"""
class PoissonFCNN(FCNN):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x, y = X[:, 0:1], X[:, 1:2]
        D = 16 * x * (1 - x) * y * (1 - y)
        return D * super().forward(X)

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