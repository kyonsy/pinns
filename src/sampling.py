import math
import numpy as np
import torch
import itertools
from collections.abc import Sequence
from numpy.typing import NDArray
from . import config as cfg

def output_grid(divisions:int) -> NDArray[np.float64]:
    """
    出力・比較用の格子点 (n+1)^2 x 2. 正方形全体 [0,1]x[0,1] を両端を含めて間隔 1/n でとる
    並びは x が外側, y が内側 (sample_grid_closed と同じ). 座標は float64 で正確に持つ
    数値解・PINN・ハード制約の関数はすべてこの点で出力し, 同じ行どうしを比べる
    """
    x, y = np.meshgrid(np.arange(divisions+1) / divisions, np.arange(divisions+1) / divisions, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel()])

def center_indices(divisions:int) -> tuple[int,int]:
    """
    中心の周りの4点の格子番号 (ceil(n/2), floor(n/2))
    4点は (i,j) をこの2つの組み合わせでとった点. n が奇数なら中心を囲む4点, 偶数なら中心1点に重なる
    """
    return math.ceil(divisions/2), math.floor(divisions/2)

def center_points(divisions:int) -> torch.Tensor:
    """
    中心の周りの4点の座標 (4,2). 並びは (a,a),(a,b),(b,a),(b,b)   (a=ceil(n/2)/n, b=floor(n/2)/n)
    境界条件 φ=1 を課す点. PINN のハード制約・ソフト制約, 数値解のすべてでこの4点を使う
    """
    a, b = (i/divisions for i in center_indices(divisions))
    return torch.tensor([[a,a],[a,b],[b,a],[b,b]],device=cfg.DEVICE)

def sample_random_default()->torch.Tensor:
    return torch.rand(cfg.NUM_SAMPLE,cfg.IN_FEATURES,device=cfg.DEVICE)

def sample_random(NUM_SAMPLE:int,IN_FEATURES:int)->torch.Tensor:
    return torch.rand(NUM_SAMPLE,IN_FEATURES,device=cfg.DEVICE)

def sample_grid(divisions:int,scale:Sequence[float])->torch.Tensor:
    """
    各次元 [0,scale[i]) の範囲の格子点をとる. 次元数は len(scale)
    点の間隔は 1/divisions で全次元共通. 各次元の点数は divisions*scale[i] 個
    例: sample_grid(100,[2.0,1.0]) → x∈[0,2) に200点, y∈[0,1) に100点
    """
    axes = [[x/divisions for x in range(round(divisions*s))] for s in scale]
    return torch.tensor(list(itertools.product(*axes)),device=cfg.DEVICE)

def sample_grid_closed(divisions:int,scale:Sequence[float])->torch.Tensor:
    """
    sample_grid と同じだが, 各次元 [0,scale[i]] の両端を含む. 出力・グラフ用
    点の間隔は 1/divisions. 各次元の点数は divisions*scale[i]+1 個
    例: sample_grid_closed(100,[1.0,1.0]) → x,y∈[0,1] に101点ずつ
    """
    axes = [[x/divisions for x in range(round(divisions*s)+1)] for s in scale]
    return torch.tensor(list(itertools.product(*axes)),device=cfg.DEVICE)

def sample_square_boundry(divisions:int) -> torch.Tensor:
    """
    角が原点に接した単位正方形の外周からサンプルをとる
    """
    bottom = torch.tensor([[x/divisions,0] for x in range(divisions)])
    right  = torch.tensor([[1,x/divisions] for x in range(divisions)])
    top    = torch.tensor([[x/divisions,1] for x in range(divisions)])
    left   = torch.tensor([[0,x/divisions] for x in range(divisions)])
    return torch.vstack([bottom,right,top,left]).to(cfg.DEVICE)






if __name__=="__main__":
    # print(sample_random_default())
    print(sample_grid(2,[1.0,1.0]))
    print(sample_square_boundry(100))
