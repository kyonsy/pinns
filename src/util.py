import os
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import torch

ArrayOrTensor = Union[torch.Tensor,npt.ArrayLike]

def _to_numpy(a:ArrayOrTensor)->np.ndarray:
    if isinstance(a,torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def save_grid(
    path: str | os.PathLike[str],
    X: ArrayOrTensor,
    Y: ArrayOrTensor,
    fmt: str = "%.6f"
) -> None:
    """2次元格子状の入力 X (N,2) と出力 Y (N,k) をgnuplot の格子形式で保存する
    ゆっくり変わる座標が一定の点を1ブロックとし、ブロック間に空行を入れる"""
    X_np = _to_numpy(X)
    Y_np = _to_numpy(Y)
    data: np.ndarray=np.hstack([X_np,Y_np])
    
    # ブロック内で値が一定の列と、1ブロックあたりの点数 n を求める
    slow: int = 0 if data[0,0] == data[1,0] else 1
    changed: np.ndarray = data[:,slow] != data[0,slow]
    n: int = int(np.argmax(changed)) if changed.any() else len(data)
    
    if len(data) % n != 0:
        raise ValueError(f"点数 {len(data)}がブロック長{n}で割り切れません")
    
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as fp:
        for block in data.reshape(-1,n,data.shape[1]):
            np.savetxt(fp,block,fmt=fmt)
            fp.write("\n")
