import sys
import numpy as np
import math as m
from numpy.typing import NDArray
from pathlib import Path
from numpy.typing import NDArray


# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config as cfg
import src.util as util

cfg.load(ROOT / "config/templete.yaml")

print(f"device: {cfg.DEVICE}")

n = cfg.GRID_DENSITY

def G(x:NDArray[np.float64],y:NDArray[np.float64]):
    a = m.ceil(cfg.GRID_DENSITY/2)/cfg.GRID_DENSITY
    b = m.floor(cfg.GRID_DENSITY/2)/cfg.GRID_DENSITY
    Dc=16 * a * (1 - a) * b * (1 - b)
    D = 16 * x * (1 - x) * y * (1 - y)
    return D/Dc


x, y = np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing="ij")
f = G(x,y)

G_out= np.column_stack([x.ravel(), y.ravel(), f.ravel()])
util.save_grid("./out/out_poisson_g.dat",G_out[:,:2],G_out[:,2:])