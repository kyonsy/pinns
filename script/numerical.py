# ポアソンの方程式を差分法で解き, PINN と比べるための数値解を保存する
# 境界条件は PINN (script/poisson.py) と同じ. 格子の細かさは config/poisson.yaml の GRID_DENSITY
# 出力: out/numerical.dat (1列目 x, 2列目 y, 3列目 φ)

import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config as cfg
import src.numerical as numerical
import src.util as util

cfg.load(ROOT / "config/numerical.yaml")

XY = numerical.solve_jacobi(cfg.GRID_DENSITY)
# XY = numerical.solve_scipy(cfg.GRID_DENSITY)   # scipy で一度に解く場合 (速い)

# solve_* は x,y∈[0,1] の (n+1)^2 点を返すが, PINN 側 (sampling.sample_grid) は x,y∈[0,1) の n^2 点なので
# x=1, y=1 の外周 (φ=0 の境界) を落として座標をそろえる
XY = XY[(XY[:,0] < 1) & (XY[:,1] < 1)]

util.save_grid(ROOT / "out/numerical.dat",XY[:,:2],XY[:,2:])

print("complete")
