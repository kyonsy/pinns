import torch
import itertools
from collections.abc import Sequence
from . import config as cfg

def sample_random_default()->torch.Tensor:
    return torch.rand(cfg.NUM_SAMPLE,cfg.IN_FEATURES,device=cfg.DEVICE)

def sample_random(NUM_SAMPLE:int,IN_FEATURES:int)->torch.Tensor:
    return torch.rand(NUM_SAMPLE,IN_FEATURES,device=cfg.DEVICE)

"""
各次元 [0,scale[i]) の範囲の格子点をとる. 次元数は len(scale)
点の間隔は 1/density で全次元共通. 各次元の点数は density*scale[i] 個
例: sample_grid(100,[2.0,1.0]) → x∈[0,2) に200点, y∈[0,1) に100点
"""
def sample_grid(density:int,scale:Sequence[float])->torch.Tensor:
    axes = [[x/density for x in range(round(density*s))] for s in scale]
    return torch.tensor(list(itertools.product(*axes)),device=cfg.DEVICE)

"""
角が原点に接した単位正方形の外周からサンプルをとる
"""
def sample_square_boundry(density:int) -> torch.Tensor:  
    bottom = torch.tensor([[x/density,0] for x in range(density)])
    right  = torch.tensor([[1,x/density] for x in range(density)])
    top    = torch.tensor([[x/density,1] for x in range(density)])
    left   = torch.tensor([[0,x/density] for x in range(density)])
    return torch.vstack([bottom,right,top,left]).to(cfg.DEVICE)






if __name__=="__main__":
    # print(sample_random_default())
    print(sample_grid(2,[1.0,1.0]))
    print(sample_square_boundry(100))
