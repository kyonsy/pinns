import math

import numpy as np
import torch

from .config import DEVICE

def output_grid(divisions:int) -> np.ndarray:
    """
    出力・比較用の格子点 (n+1)^2 x 2. [0,1]x[0,1] を両端を含めて間隔 1/n でとる (float64)
    並びは x が外側, y が内側. 数値解・PINN・ハード制約の関数はすべてこの点で出力し, 同じ行どうしを比べる
    """
    x, y = np.meshgrid(np.arange(divisions+1) / divisions, np.arange(divisions+1) / divisions, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel()])

def center_indices(divisions:int) -> tuple[int,int]:
    """
    中心の4点の格子番号 (ceil(n/2), floor(n/2)). 4点は (i,j) をこの2つの組み合わせでとった点
    n が奇数なら中心を囲む4点, 偶数なら中心の1点に重なる
    """
    return math.ceil(divisions/2), math.floor(divisions/2)

def center_points(divisions:int) -> torch.Tensor:
    """中心の4点の座標 (4,2). 並びは (a,a),(a,b),(b,a),(b,b)   (a=ceil(n/2)/n, b=floor(n/2)/n)"""
    i, j = center_indices(divisions)
    a, b = i/divisions, j/divisions
    return torch.tensor([[a,a],[a,b],[b,a],[b,b]],device=DEVICE)

def sample_interior(divisions:int) -> torch.Tensor:
    """
    ラプラス方程式の残差を計算する点 (学習用, float32). 間隔 1/n の格子点のうち,
    外周と中心の4点 (境界条件で値が決まっている点) を除いたもの. 差分法の数値解も, この点でだけ方程式を課している
    中心の4点では正しい解でも Δφ=0 にならない (φ がそこで尖っている) ので, 含めると無理な条件になる
    """
    n = divisions
    c1, c2 = center_indices(n)
    centers = [(c1,c1), (c1,c2), (c2,c1), (c2,c2)]
    points = [[i/n, j/n] for i in range(1, n) for j in range(1, n) if (i, j) not in centers]
    return torch.tensor(points,device=DEVICE)

def sample_square_boundry(divisions:int) -> torch.Tensor:
    """[0,1]x[0,1] の外周の点"""
    bottom = torch.tensor([[x/divisions,0] for x in range(divisions)])
    right  = torch.tensor([[1,x/divisions] for x in range(divisions)])
    top    = torch.tensor([[x/divisions,1] for x in range(divisions)])
    left   = torch.tensor([[0,x/divisions] for x in range(divisions)])
    return torch.vstack([bottom,right,top,left]).to(DEVICE)
