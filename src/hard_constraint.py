import torch
import torch.nn as nn
import numpy as np
import math as m

from . import config as cfg

n = cfg.GRID_DENSITY
a = m.ceil(n/2)/n
b = m.floor(n/2)/n
P_k = torch.tensor([[a,a],[a,b],[b,a],[b,b]])
h = 1/float(n)
s2 = h**2
r0 = h / m.sqrt(2)

"""
境界条件用
φ = G + D*C*N と使う
    D = 16x(1-x)y(1-y)          : 外周で0, 中心で1
    G = D/D(中心の4点)           : 外周で0, 中心の4点で1
    C = Π_k (1-exp(-|X-p_k|²/s²)) : 中心の4点 p_k で0, 離れると1
"""
    
def D(X:torch.Tensor)->torch.Tensor:
    """外周で 0 になる関数 D"""
    x, y = X[:, 0:1], X[:, 1:2]
    return 16 * x * (1 - x) * y * (1 - y)

def G(X:torch.Tensor) ->torch.Tensor:
    return D(X) / D(P_k[0])

def C(X:torch.Tensor):
    r2 = ((X[:, None, :] - P_k[None, :, :])**2).sum(dim=2) # (N,4)
    return (1 - torch.exp(-r2 / s2)).prod(dim=1, keepdim=True)


def L(X:torch.Tensor):
    x, y = X[:, 0:1], X[:, 1:2]
    return torch.log(x**2+y**2) / torch.log(2 * torch.tensor(a)**2)