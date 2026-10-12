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

n = cfg.GRID_DENSITY


x, y = np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing="ij")
X =np.column_stack([x.ravel(), y.ravel()])

hard      = np.loadtxt("./out/out_poisson_hard.dat")[:,2:]
g         = np.loadtxt("./out/out_poisson_hard.dat")[:,2:]
numerical = np.loadtxt("./out/out_poisson_numerical.dat")[:,2:]

