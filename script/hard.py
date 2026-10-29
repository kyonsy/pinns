"""
ハード制約の PINN でポアソンの方程式を解き, 結果を out/ に保存する
使い方: python -m script.hard <モデル> <設定ファイル> <出力ファイル>
    例: python -m script.hard g  hard.yaml hard_g.dat    (φ = G + D·C·N)
        python -m script.hard gl hard.yaml hard_gl.dat   (φ = G·L + D·C·N)
"""
import argparse

import torch

import src.config as config
import src.fcnn as fcnn
import src.sampling as sampling
import src.train as train
import src.util as util

# モデルの名前 -> クラス
MODELS = {
    "g"  : fcnn.PoissonFCNN1,   # φ = G + D·C·N
    "gl" : fcnn.PoissonFCNN2,   # φ = G·L + D·C·N
}

def main() -> None:
    parser = argparse.ArgumentParser(description="ハード制約の PINN")
    parser.add_argument("model", choices=MODELS, help="境界条件を満たす部分 (g: G, gl: G·L)")
    parser.add_argument("config", help="config/ の設定ファイル名 (例: hard.yaml)")
    parser.add_argument("output", help="out/ に保存するファイル名 (例: hard_g.dat)")
    args = parser.parse_args()

    cfg = config.load(config.ROOT / "config" / args.config)
    print(f"device: {config.DEVICE}")

    model = MODELS[args.model](cfg).to(config.DEVICE)
    train.train_hard(model,cfg)

    # 出力は数値解と同じ点 (sampling.output_grid) でとる. 座標は float64 のまま保存し, モデルには float32 で入れる
    X_out = torch.tensor(sampling.output_grid(cfg.divisions()),device=config.DEVICE)
    with torch.no_grad():
        Y_out = model(X_out.float())
    util.save_grid(config.ROOT / "out" / args.output,X_out,Y_out)
    print("complete")

if __name__ == "__main__":
    main()
