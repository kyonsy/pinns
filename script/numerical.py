"""
ポアソンの方程式を差分法で解き, PINN と比べるための数値解を out/numerical.dat に保存する
境界条件は PINN と同じ. 格子の細かさは config/numerical.yaml の POINTS_PER_SIDE
出力: 1列目 x, 2列目 y, 3列目 φ (点は sampling.output_grid と同じ)
使い方: python -m script.numerical [--solver jacobi|scipy] [--repeat N]
"""
import argparse

import src.config as config
import src.numerical as numerical
import src.util as util

# 解き方の名前 -> 関数
SOLVERS = {
    "jacobi" : numerical.solve_jacobi,   # ヤコビ法 (C 版 csrc/poisson_jacobi.c と同じ)
    "scipy"  : numerical.solve_scipy,    # scipy で一度に解く (速い)
}

def main() -> None:
    parser = argparse.ArgumentParser(description="差分法による数値解")
    parser.add_argument("--solver", default="jacobi", choices=SOLVERS, help="解き方")
    parser.add_argument("--repeat", type=int, default=1, help="同じ計算を繰り返す回数 (C 版との速度比較用)")
    args = parser.parse_args()
    run(args.solver, args.repeat)

def run(solver:str="jacobi", repeat:int=1) -> None:
    cfg = config.load(config.ROOT / "config/numerical.yaml")

    for _ in range(repeat):
        XY = SOLVERS[solver](cfg.divisions())

    util.save_grid(config.ROOT / "out/numerical.dat",XY[:,:2],XY[:,2:])
    print("complete")

if __name__ == "__main__":
    main()
