# ポアソンの方程式を差分法で解き, PINN と比べるための数値解を保存する
# 境界条件は PINN (script/poisson_eq.py) と同じ. 格子の細かさは config/poisson_eq.yaml の grid_density
# 出力: out/numerical.dat (1列目 x, 2列目 y, 3列目 φ)

import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import src.config as cfg
import src.numerical as numerical

cfg.load(ROOT / "config/poisson_eq.yaml")

XY = numerical.solve_poisson_jacobi(cfg.grid_density)
# XY = numerical.solve_poisson_scipy(cfg.grid_density)   # scipy で一度に解く場合 (速い)

np.savetxt(ROOT / "out/numerical.dat", XY, "%.6f")
print("complete")
