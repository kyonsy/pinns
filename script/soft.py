"""
ソフト制約の PINN でポアソンの方程式を解き, 結果を out/ に保存する
境界条件は損失関数で課すので, 出力を変換しない素の FCNN を使う
使い方: python -m script.soft <設定ファイル> <出力ファイル> [--seed N]   (パスはプロジェクトのルートから)
    例: python -m script.soft config/soft_w1.yaml out/soft_w1.dat
        python -m script.soft config/soft_w1.yaml out/soft_w1_seed1.dat --seed 1   (シードだけ変えて実行)
"""
import os
# GPU でも同じシードなら毎回同じ結果にするための設定. torch が GPU を使い始める前に設定する
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import dataclasses
from datetime import datetime

import torch

import src.config as config
import src.fcnn as fcnn
import src.sampling as sampling
import src.train as train
import src.util as util

def main() -> None:
    parser = argparse.ArgumentParser(description="ソフト制約の PINN")
    parser.add_argument("config", help="設定ファイル. プロジェクトのルートからのパス (例: config/soft_w1.yaml)")
    parser.add_argument("output", help="保存するファイル. プロジェクトのルートからのパス (例: out/soft_w1.dat)")
    parser.add_argument("--seed", type=int, help="乱数のシード. 省略すると設定ファイルの SEED")
    args = parser.parse_args()

    cfg = config.load(config.ROOT / args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, SEED=args.seed)

    device_name = torch.cuda.get_device_name() if config.DEVICE.type == "cuda" else "cpu"
    print(f"device: {device_name}  seed: {cfg.SEED}")
    torch.manual_seed(cfg.SEED)                 # 重みの初期値を毎回同じにする
    torch.use_deterministic_algorithms(True)    # 実行のたびに結果が変わる計算方法を使わない

    model = fcnn.FCNN(cfg).to(config.DEVICE)
    train.train_soft(model,cfg)

    # 出力は数値解と同じ点 (sampling.output_grid) でとる. 座標は float64 のまま保存し, モデルには float32 で入れる
    X_out = torch.tensor(sampling.output_grid(cfg.divisions()),device=config.DEVICE)
    with torch.no_grad():
        Y_out = model(X_out.float())
    # 先頭の行に, どの設定・シード・GPU で作ったかを残す
    header = f"config={args.config} seed={cfg.SEED} device={device_name} torch={torch.__version__} date={datetime.now():%Y-%m-%d %H:%M}"
    util.save_grid(config.ROOT / args.output,X_out,Y_out,header=header)
    print("complete")

if __name__ == "__main__":
    main()
