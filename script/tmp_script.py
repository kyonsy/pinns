import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


import numpy as np
import src.util as util

f1=ROOT / "out/out_hard.dat"
# f2=ROOT / "out/out_numerical.dat"
f3=ROOT / "out/out_soft1.dat"
f4=ROOT / "out/out_soft2.dat"
f5=ROOT / "out/out_soft3.dat"

fs = {f1,f3,f4,f5}

for f in fs:
    nf = np.loadtxt(f)
    util.save_grid(f,nf[:,:2],nf[:,2:])
    

