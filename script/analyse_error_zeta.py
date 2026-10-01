# 減衰比ζが0.0~2.0の場合について、[0,2π]内での2乗平均平方根誤差(RMSE)と減衰比との関係を調べる
# 入力: out/zeta<ζ>.dat (script/dump_sys_zeta.py の出力, 学習区間は[0,2π]で固定)
# 出力: out/rmse_zeta.dat (1列目 ζ, 2列目 RMSE)

import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import math as m
import src.config as cfg

cfg.load(ROOT / "config/dump_sys.yaml")

zeta_max = 2.0
zeta_step = 0.1

num_line = round(2 * m.pi * cfg.plot_density)

zetas = [i * zeta_step for i in range(round(zeta_max / zeta_step) + 1)]

results: list[list[float]] = []
for zeta in zetas:
    nerrors = np.loadtxt(ROOT / f"out/zeta{zeta:.1f}.dat", usecols=3)[:num_line]
    rmse = float(np.sqrt(np.mean(nerrors**2)))
    results.append([zeta, rmse])

np.savetxt(ROOT / "out/rmse_zeta.dat", results, fmt="%.6e")
