# 学習区間が[0,2π]~[0,5π]の場合について、[0,2π]内での平均2乗誤差と学習区間との関係を調べる




import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import math as m
import src.config as cfg

cfg.load(ROOT / "config/dump_sys.yaml")

range_start = 4
range_max = 20

num_line=round(2 * m.pi * cfg.plot_density)

print(num_line) 
tranges = np.array([[x+range_start] for x in range(range_max-range_start+1)])
# print(tranges)

results :list[list[float]] = []
for [x] in tranges:
    trange = m.pi/2 * x
    nerrors = np.loadtxt(f"./out/out{x}.dat", usecols=3)[:num_line]
    rmse = np.sqrt(np.mean(nerrors**2))
    results.append([trange, rmse])

np.savetxt("./out/rmse_trange.dat", results, fmt="%.6e")



    
    
        
