from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from functools import partial

import torch
import torch.nn as nn
import yaml

# プロジェクトのルート (src/ の1つ上). config/ や out/ はここからの相対パスで指す
ROOT          : Path = Path(__file__).resolve().parent.parent

# 計算に使うデバイス. GPU(CUDA)が使えればGPU, なければCPU (yaml では変えない定数)
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
    "POINTS_PER_SIDE": int,          
    "LEARNING_RATE" : float,   
    "LOSS_WEIGHTS"  : partial(torch.tensor,dtype=torch.float64),  
    "ACTIVATION"    : _activation,
}

@dataclass(frozen=True)
class Config:
    """
    yaml 1つ分の設定. load() で作り, 使う関数に引数で渡す. 作ったあとは値を変更できない
    yaml に書かれていない項目は None になる
    """
    IN_FEATURES    : int | None             = None  # 入力層の幅
    OUT_FEATURES   : int | None             = None  # 出力層の幅
    NUM_LAYERS     : int | None             = None  # 隠れ層の数
    WIDTH          : int | None             = None  # 隠れ層の幅
    EPOCH          : int | None             = None  # 学習回数
    POINTS_PER_SIDE: int | None             = None  # 1辺 [0,1] にとる格子点の数 (両端を含む). 間隔は 1/(N-1). 中心の4点を作るため偶数にする
    LEARNING_RATE  : float | None           = None  # 学習率
    LOSS_WEIGHTS   : torch.Tensor | None    = None  # 損失関数の各項の重み [方程式, 外周, 中心4点]
    ACTIVATION     : type[nn.Module] | None = None  # 活性化関数

    def divisions(self) -> int:
        """
        格子の分割数 n (= POINTS_PER_SIDE - 1). 点の間隔は 1/n
        sampling の関数や数値解はこの分割数を受け取る
        """
        return self.POINTS_PER_SIDE - 1

def from_dict(data:dict[str, Any]) -> Config:
    """yaml と同じ形の辞書 {変数名: 値} から Config を作る. 値は _CONVERT で変換する"""
    values: dict[str, Any] = {}
    for key, value in data.items():
        if key not in _CONVERT:
            raise KeyError(f"yamlに未知の変数があります: {key}")
        if value is None:
            raise ValueError(f"yamlで {key} の値がからです")
        values[key] = _CONVERT[key](value)
    return Config(**values)

def load(path: str | Path) -> Config:
    """yamlファイルを読んで Config を作る"""
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return from_dict(data)
