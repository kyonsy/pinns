"""
ソフト制約の PINN でポアソンの方程式を解き, 結果を out/ に保存する
引数は config/ の設定ファイル名. 出力ファイル名は src/experiments.py の一覧から決まる
一覧にない設定ファイル (test.yaml など) は --output で出力ファイル名を指定する
使い方: python -m script.soft [soft_w1.yaml|soft_w1e2.yaml|...] [--output <ファイル名>]

境界条件は損失関数で課すので, 出力を変換しない素の FCNN を使う
"""
import argparse

import src.config as config
import src.experiments as experiments
import src.fcnn as fcnn
import src.train as train
import src.util as util

def run(config_file:str, output:str) -> None:
    """config/<config_file> の設定で学習し, out/<output> に保存する"""
    cfg = config.load(config.ROOT / "config" / config_file)
    print(f"device: {config.DEVICE}")

    model = fcnn.FCNN(cfg).to(config.DEVICE)
    train.train_soft(model,cfg)
    util.save_prediction(config.ROOT / "out" / output,model,cfg.divisions())
    print("complete")

def main() -> None:
    parser = argparse.ArgumentParser(description="ソフト制約の PINN")
    parser.add_argument("config", nargs="?", default="soft_w1.yaml",
                        help=f"config/ の設定ファイル名 (一覧: {', '.join(experiments.of_kind('soft'))})")
    parser.add_argument("--output", help="out/ に保存するファイル名. 省略すると一覧の出力ファイル名")
    args = parser.parse_args()

    exp = experiments.EXPERIMENTS.get(args.config)
    output = args.output or (exp.output if exp else None)
    if output is None:
        parser.error(f"{args.config} は実験の一覧にないので --output で出力ファイル名を指定してください")
    print(f"{args.config} -> {output}: {exp.description if exp else '(一覧にない設定)'}")
    run(args.config, output)

if __name__ == "__main__":
    main()
