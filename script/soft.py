"""
ソフト制約の PINN でポアソンの方程式を解き, out/<name>.dat に保存する
name は src/experiments.py の実験の名前. 一覧にない名前なら config/<name>.yaml を読む (test など)
使い方: python -m script.soft [soft_w1|soft_w1e2|...]

境界条件は損失関数で課すので, 出力を変換しない素の FCNN を使う
"""
import argparse

import src.config as config
import src.experiments as experiments
import src.fcnn as fcnn
import src.train as train
import src.util as util

def run(name:str) -> None:
    exp = experiments.EXPERIMENTS.get(name)
    cfg = config.load(config.ROOT / f"config/{exp.config if exp else name}.yaml")
    print(f"{name}: {exp.description if exp else '(一覧にない設定)'}")
    print(f"device: {config.DEVICE}")

    model = fcnn.FCNN(cfg).to(config.DEVICE)
    train.train_soft(model,cfg)
    util.save_prediction(config.ROOT / f"out/{name}.dat",model,cfg.divisions())
    print("complete")

def main() -> None:
    parser = argparse.ArgumentParser(description="ソフト制約の PINN")
    parser.add_argument("name", nargs="?", default="soft_w1",
                        help=f"実験の名前 ({', '.join(experiments.of_kind('soft'))}) か config/ の yaml の名前")
    args = parser.parse_args()
    run(args.name)

if __name__ == "__main__":
    main()
