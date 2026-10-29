"""
ハード制約の PINN でポアソンの方程式を解き, out/<name>.dat に保存する
name は src/experiments.py の実験の名前. 設定は config/hard.yaml
使い方: python -m script.hard [hard_g|hard_gl]
"""
import argparse

import src.config as config
import src.experiments as experiments
import src.fcnn as fcnn
import src.train as train
import src.util as util

# 実験の名前 -> モデルのクラス
MODELS = {
    "hard_g"  : fcnn.PoissonFCNN1,
    "hard_gl" : fcnn.PoissonFCNN2,
}

def run(name:str) -> None:
    exp = experiments.EXPERIMENTS[name]
    cfg = config.load(config.ROOT / f"config/{exp.config}.yaml")
    print(f"{name}: {exp.description}")
    print(f"device: {config.DEVICE}")

    model = MODELS[name](cfg).to(config.DEVICE)
    train.train_hard(model,cfg)
    util.save_prediction(config.ROOT / f"out/{name}.dat",model,cfg.divisions())
    print("complete")

def main() -> None:
    parser = argparse.ArgumentParser(description="ハード制約の PINN")
    parser.add_argument("name", nargs="?", default="hard_g", choices=experiments.of_kind("hard"), help="実験の名前")
    args = parser.parse_args()
    run(args.name)

if __name__ == "__main__":
    main()
