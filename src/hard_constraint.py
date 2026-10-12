import torch
import math as m
from functools import cache

from . import config as cfg




"""
境界条件用
φ = G + D*C*N と使う
    D = 16x(1-x)y(1-y)          : 外周で0, 中心で1
    G = D/D(中心の4点)           : 外周で0, 中心の4点で1
    C = Π_k (1-exp(-|X-p_k|²/s²)) : 中心の4点 p_k で0, 離れると1
"""
    
@cache
def param():
    n = cfg.GRID_DENSITY
    a,b = m.ceil(n/2)/n,m.floor(n/2)/n
    h = 1/n
    P_k = torch.tensor([[a,a],[a,b],[b,a],[b,b]])
    return P_k,h**2,h**2/2
    
@cache
def D_p() -> torch.Tensor:
    return D(param()[0][:1])    
    

def D(X:torch.Tensor)->torch.Tensor:
    """外周で 0 になる関数 D"""
    x, y = X[:, 0:1], X[:, 1:2]
    return 16 * x * (1 - x) * y * (1 - y)

def G(X:torch.Tensor) ->torch.Tensor:
    return D(X) / D_p()

def C(X:torch.Tensor):
    P_k,s2,_ = param()
    r2 = ((X[:, None, :] - P_k[None, :, :])**2).sum(dim=2) # (N,4)
    return (1 - torch.exp(-r2 / s2)).prod(dim=1, keepdim=True)

def L(X:torch.Tensor):
    _,_,r0_2 = param()
        
    x, y = X[:, 0:1], X[:, 1:2]
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    return torch.log(r2 + 1e-12) / m.log(r0_2)