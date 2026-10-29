# ポアソンの方程式を差分法で解き, PINN と比べるための数値解を保存する
# 境界条件は PINN (script/poisson.py) と同じ. 格子の細かさは config/numerical.yaml の POINTS_PER_SIDE
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
for _ in range(100):
    XY = numerical.solve_jacobi(cfg.divisions())
    
# XY = numerical.solve_scipy(cfg.divisions())   # scipy で一度に解く場合 (速い)

# solve_* は x,y∈[0,1] の (n+1)^2 点を返す. PINN 側の出力 (sampling.sample_grid_closed) と同じ点なのでそのまま保存する

util.save_grid(ROOT / "out/numerical.dat",XY[:,:2],XY[:,2:])

print("complete")
