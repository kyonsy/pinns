from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

in_features: int = 2          # 入力層の幅
out_features: int = 1         # 出力層の幅
num_layers: int = 4           # 隠れ層の数
width: int = 50               # 隠れ層の幅
epoch:int=10000               # 学習回数
num_sample:int=100            # ランダム・境界サンプリングの点数
num_plot:int=100              # 解析解プロットの点数
grid_density:int=100          # 学習用グリッドの単位長さあたりの点数
plot_density:int=100          # 出力用グリッドの単位長さあたりの点数
learning_rate:float = 1e-3    # 学習率

# 計算に使うデバイス. GPU(CUDA)が使えればGPU, なければCPU
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_weights:torch.Tensor =torch.tensor([1.])

Activation: nn.Module = nn.Tanh()

# yaml で活性化関数を名前指定するための対応表
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

"""
yamlファイルの値でこのモジュールの設定値を上書きする
yamlに書かれていない項目はデフォルト値のまま
"""
def load(path: str | Path) -> None:
    global in_features, out_features, num_layers, width, epoch
    global num_sample, num_plot, grid_density, plot_density, learning_rate, loss_weights, Activation, device

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    in_features   = int(data.get("in_features", in_features))
    out_features  = int(data.get("out_features", out_features))
    num_layers    = int(data.get("num_layers", num_layers))
    width         = int(data.get("width", width))
    epoch         = int(data.get("epoch", epoch))
    num_sample    = int(data.get("num_sample", num_sample))
    num_plot      = int(data.get("num_plot", num_plot))
    grid_density  = int(data.get("grid_density", grid_density))
    plot_density  = int(data.get("plot_density", plot_density))
    learning_rate = float(data.get("learning_rate", learning_rate))

    if "loss_weights" in data:
        loss_weights = torch.tensor(data["loss_weights"], dtype=torch.float32)
    if "activation" in data:
        Activation = ACTIVATIONS[data["activation"].lower()]()
    # device: auto(デフォルト) / cuda / cpu
    if data.get("device", "auto") != "auto":
        device = torch.device(data["device"])

    unknown: set[str] = set(data) - {
        "in_features", "out_features", "num_layers", "width", "epoch",
        "num_sample", "num_plot", "grid_density", "plot_density",
        "learning_rate", "loss_weights", "activation", "device",
    }
    if unknown:
        raise KeyError(f"{path} に未知の設定項目があります: {sorted(unknown)}")
