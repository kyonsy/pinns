"""
ソフト制約の PINN でポアソンの方程式を解き, 結果を out/ に保存する
境界条件は損失関数で課すので, 出力を変換しない素の FCNN を使う
使い方: python -m script.soft <設定ファイル> <出力ファイル>
    例: python -m script.soft soft_w1.yaml soft_w1.dat   (config/soft_w1.yaml を読み, out/soft_w1.dat に保存)
"""
import argparse

import torch

import src.config as config
import src.fcnn as fcnn
import src.sampling as sampling
import src.train as train
import src.util as util

def main() -> None:
    parser = argparse.ArgumentParser(description="ソフト制約の PINN")
    parser.add_argument("config", help="config/ の設定ファイル名 (例: soft_w1.yaml)")
    parser.add_argument("output", help="out/ に保存するファイル名 (例: soft_w1.dat)")
    args = parser.parse_args()

    cfg = config.load(config.ROOT / "config" / args.config)
    print(f"device: {config.DEVICE}")

    model = fcnn.FCNN(cfg).to(config.DEVICE)
    train.train_soft(model,cfg)

    # 出力は数値解と同じ点 (sampling.output_grid) でとる. 座標は float64 のまま保存し, モデルには float32 で入れる
    X_out = torch.tensor(sampling.output_grid(cfg.divisions()),device=config.DEVICE)
    with torch.no_grad():
        Y_out = model(X_out.float())
    util.save_grid(config.ROOT / "out" / args.output,X_out,Y_out)
    print("complete")

if __name__ == "__main__":
    main()
