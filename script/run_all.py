"""
数値解, src/experiments.py の BATCH の実験, ハード制約の関数, 差と MSE をまとめて計算して out/ に保存する
設定は Config で引数として渡しているので, 1つのプロセスで続けて実行できる
使い方: python -m script.run_all
"""
import src.experiments as experiments
from . import export_hard_constraint, hard, make_diff, numerical, soft

# 実験の種類 -> 学習して保存する関数
RUNNERS = {
    "soft" : soft.run,
    "hard" : hard.run,
}

def main() -> None:
    numerical.run()
    for name in experiments.BATCH:
        RUNNERS[experiments.EXPERIMENTS[name].kind](name)
    export_hard_constraint.run()
    make_diff.run()

if __name__ == "__main__":
    main()
