from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# プロジェクトのルート (src/ の1つ上). config/ や out/ はここからの相対パスで指す
ROOT = Path(__file__).resolve().parent.parent

# 計算に使うデバイス. GPU(CUDA)が使えればGPU, なければCPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# yaml で活性化関数を名前で指定するための対応表
ACTIVATIONS = {
    "tanh"    : nn.Tanh,
    "relu"    : nn.ReLU,
    "sigmoid" : nn.Sigmoid,
    "gelu"    : nn.GELU,
    "silu"    : nn.SiLU,
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
    SEED           : int | None             = None  # 乱数のシード

    def divisions(self) -> int:
        """格子の分割数 n (= POINTS_PER_SIDE - 1). 点の間隔は 1/n"""
        return self.POINTS_PER_SIDE - 1

def load(path: str | Path) -> Config:
    """
    yamlファイルを読んで Config を作る
    yaml に Config にない項目があるとエラー (TypeError: unexpected keyword argument) になる
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # yaml のままでは使えない値を変換する
    if "ACTIVATION" in data:
        data["ACTIVATION"] = ACTIVATIONS[data["ACTIVATION"]]
    if "LOSS_WEIGHTS" in data:
        data["LOSS_WEIGHTS"] = torch.tensor(data["LOSS_WEIGHTS"], dtype=torch.float64)
    if "LEARNING_RATE" in data:
        data["LEARNING_RATE"] = float(data["LEARNING_RATE"])   # 1e-3 のように書くと yaml では文字列になるため

    return Config(**data)
