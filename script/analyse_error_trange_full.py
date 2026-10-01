# 学習区間が[0,π/2]~[0,10π]の場合について、学習区間全体[0,T]での2乗平均平方根誤差(RMSE)と学習区間との関係を調べる
# (analyse_error_trange.py は [0,2π] 内だけで比べたもの. こちらは各学習区間の全体で比べる)
# 入力: out/out<j>.dat (script/dump_sys_trange.py の出力. j 番目は学習区間 [0, jπ/2], 区間全体のデータが入っている)
# 出力: out/rmse_trange_full.dat (1列目 学習区間の終端 T, 2列目 RMSE)

import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import math as m

range_start = 1
range_max = 20

results: list[list[float]] = []
for x in range(range_start, range_max + 1):
    trange = m.pi/2 * x
    # ファイルには [0,T] 全体の誤差が入っているので, 全行を使う
    nerrors = np.loadtxt(ROOT / f"out/out{x}.dat", usecols=3)
    rmse = float(np.sqrt(np.mean(nerrors**2)))
    results.append([trange, rmse])

np.savetxt(ROOT / "out/rmse_trange_full.dat", results, fmt="%.6e")
