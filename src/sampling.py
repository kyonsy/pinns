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
点の間隔は 1/divisions で全次元共通. 各次元の点数は divisions*scale[i] 個
例: sample_grid(100,[2.0,1.0]) → x∈[0,2) に200点, y∈[0,1) に100点
"""
def sample_grid(divisions:int,scale:Sequence[float])->torch.Tensor:
    axes = [[x/divisions for x in range(round(divisions*s))] for s in scale]
    return torch.tensor(list(itertools.product(*axes)),device=cfg.DEVICE)

"""
sample_grid と同じだが, 各次元 [0,scale[i]] の両端を含む. 出力・グラフ用
点の間隔は 1/divisions. 各次元の点数は divisions*scale[i]+1 個
例: sample_grid_closed(100,[1.0,1.0]) → x,y∈[0,1] に101点ずつ
"""
def sample_grid_closed(divisions:int,scale:Sequence[float])->torch.Tensor:
    axes = [[x/divisions for x in range(round(divisions*s)+1)] for s in scale]
    return torch.tensor(list(itertools.product(*axes)),device=cfg.DEVICE)

"""
角が原点に接した単位正方形の外周からサンプルをとる
"""
def sample_square_boundry(divisions:int) -> torch.Tensor:  
    bottom = torch.tensor([[x/divisions,0] for x in range(divisions)])
    right  = torch.tensor([[1,x/divisions] for x in range(divisions)])
    top    = torch.tensor([[x/divisions,1] for x in range(divisions)])
    left   = torch.tensor([[0,x/divisions] for x in range(divisions)])
    return torch.vstack([bottom,right,top,left]).to(cfg.DEVICE)






if __name__=="__main__":
    # print(sample_random_default())
    print(sample_grid(2,[1.0,1.0]))
    print(sample_square_boundry(100))
