"""
gnuplot で out/ の結果のグラフをまとめて描く. 2d は graph/2d/, 3d は graph/3d/ に保存する
描く結果は src/experiments.py の一覧. 先に script.run_all で out/ を作っておく
使い方: python -m script.plot_all
"""
import subprocess

import src.config as config
import src.experiments as experiments

def gnuplot(plt:str, *args:str, n:int|None=None) -> None:
    """plot/<plt> を引数 args で実行する. n を渡すと gnuplot の変数 n (分割数) に入れる"""
    cmd = ["gnuplot"]
    if n is not None:
        cmd += ["-e", f"n={n}"]
    cmd += ["-c", f"./plot/{plt}", *args]
    subprocess.run(cmd, check=True, cwd=config.ROOT)

def main() -> None:
    # 中心の4点や y=0.5 の行の位置を出すのに, 出力格子の分割数を gnuplot に渡す
    n = config.load(config.ROOT / "config/templete.yaml").divisions()

    # gnuplot は保存先のフォルダを作らないので先に作っておく
    for d in ["2d", "3d"]:
        (config.ROOT / "graph" / d).mkdir(parents=True, exist_ok=True)

    for name in experiments.SURFACES:
        gnuplot("2d/contour.plt", name)
        gnuplot("3d/surface.plt", name)

    for a, b in experiments.DIFF_PAIRS:
        gnuplot("2d/diff.plt", experiments.diff_name(a, b))
        gnuplot("3d/diff.plt", experiments.diff_name(a, b))

    # y=0.5 の断面
    for names in experiments.SECTIONS:
        gnuplot("2d/section.plt", *names, n=n)

    # ハード制約の説明図 (G の等高線図と中心付近の C, G の断面)
    gnuplot("2d/hard_constraint.plt", n=n)
    gnuplot("3d/hard_constraint.plt", n=n)

    # ソフト制約の断面と, 数値解との MSE の棒グラフ
    gnuplot("2d/section_soft.plt", "numerical", *experiments.SOFT_WEIGHTS, n=n)
    gnuplot("2d/mse.plt")
    print("complete")

if __name__ == "__main__":
    main()
