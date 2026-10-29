"""
ハード制約の PINN でポアソンの方程式を解き, 結果を out/ に保存する
引数は config/ の設定ファイル名. 出力ファイル名は src/experiments.py の一覧から決まる
使い方: python -m script.hard [hard_g.yaml|hard_gl.yaml]
"""
import argparse

import src.config as config
import src.experiments as experiments
import src.fcnn as fcnn
import src.train as train
import src.util as util

# 設定ファイル名 -> モデルのクラス
MODELS = {
    "hard_g.yaml"  : fcnn.PoissonFCNN1,
    "hard_gl.yaml" : fcnn.PoissonFCNN2,
}

def run(config_file:str) -> None:
    """config/<config_file> の設定で学習し, 一覧の出力ファイルに保存する"""
    exp = experiments.EXPERIMENTS[config_file]
    cfg = config.load(config.ROOT / "config" / config_file)
    print(f"{config_file} -> {exp.output}: {exp.description}")
    print(f"device: {config.DEVICE}")

    model = MODELS[config_file](cfg).to(config.DEVICE)
    train.train_hard(model,cfg)
    util.save_prediction(config.ROOT / "out" / exp.output,model,cfg.divisions())
    print("complete")

def main() -> None:
    parser = argparse.ArgumentParser(description="ハード制約の PINN")
    parser.add_argument("config", nargs="?", default="hard_g.yaml", choices=experiments.of_kind("hard"),
                        help="config/ の設定ファイル名")
    args = parser.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()
