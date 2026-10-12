from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
from typing import Any
from functools import partial

import torch
import torch.nn as nn
import yaml

IN_FEATURES   : int          # 入力層の幅
OUT_FEATURES  : int          # 出力層の幅
NUM_LAYERS    : int          # 隠れ層の数
WIDTH         : int          # 隠れ層の幅
EPOCH         : int          # 学習回数
NUM_SAMPLE    : int          # ランダム・境界サンプリングの点数
NUM_PLOT      : int          # 解析解プロットの点数
GRID_DENSITY  : int          # 学習用グリッドの単位長さあたりの点数
PLOT_DENSITY  : int          # 出力用グリッドの単位長さあたりの点数
LEARNING_RATE : float        # 学習率
LOSS_WEIGHTS  : torch.Tensor # 損失関数の各項の重み

# 活性化関数
ACTIVATION    : type[nn.Module]      

# 計算に使うデバイス. GPU(CUDA)が使えればGPU, なければCPU
DEVICE        : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# yaml で活性化関数を名前指定するための対応表
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh"    : nn.Tanh,
    "relu"    : nn.ReLU,
    "sigmoid" : nn.Sigmoid,
    "gelu"    : nn.GELU,
    "silu"    : nn.SiLU,
}

def _activation(name:str)->type[nn.Module]:
    if name not in _ACTIVATIONS:
        raise ValueError(f"未対応の活性化関数:{name} (候補: {list(_ACTIVATIONS)})")
    return _ACTIVATIONS[name]

# 変数名 -> 変数変換
_CONVERT:dict[str,Callable[[Any],Any]] = {
    "IN_FEATURES"   : int,           
    "OUT_FEATURES"  : int,          
    "NUM_LAYERS"    : int,            
    "WIDTH"         : int,                 
    "EPOCH"         : int,                  
    "NUM_SAMPLE"    : int,             
    "NUM_PLOT"      : int,               
    "GRID_DENSITY"  : int,           
    "PLOT_DENSITY"  : int,           
    "LEARNING_RATE" : float,   
    "LOSS_WEIGHTS"  : partial(torch.tensor,dtype=torch.float64),  
    "ACTIVATION"    : _activation      
}

"""
yamlファイルの値でこのモジュールの設定値を上書きする
yamlに書かれていない項目はデフォルト値のまま
"""
def load(path: str | Path) -> None:
   
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    for key, value in data.items():
        if key not in _CONVERT:
            raise KeyError(f"yamlに未知の変数があります: {key}")
        if value is None:
            raise ValueError(f"yamlで {key} の値がからです")
        globals()[key] = _CONVERT[key](value)

def __getattr__(name:str):
    if name in _CONVERT:
        raise AttributeError(f"config.{name}はyamlで設定されていません")
    raise AttributeError(f"module 'config' has no attribute '{name}'")
