from pathlib import Path

import numpy as np
import torch

def _to_numpy(a) -> np.ndarray:
    if isinstance(a,torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def save_grid(path, X, Y, fmt:str="%.6f", header:str|None=None) -> None:
    """
    格子状の入力 X (N,2) と出力 Y (N,k) を gnuplot の格子形式で保存する
    X は x が外側, y が内側の並び (sampling.output_grid と同じ) とし, x が同じ点ごとに空行で区切る
    header を渡すと先頭に "# header" の行を書く (gnuplot と np.loadtxt は読み飛ばす)
    """
    data = np.hstack([_to_numpy(X), _to_numpy(Y)])
    n = int((data[:,0] == data[0,0]).sum())    # 1ブロック (x が同じ点) の点数

    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as fp:
        if header is not None:
            fp.write(f"# {header}\n")
        for block in data.reshape(-1,n,data.shape[1]):
            np.savetxt(fp,block,fmt=fmt)
            fp.write("\n")
