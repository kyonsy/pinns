import sys
import numpy as np
import torch
import math as m
from numpy.typing import NDArray
from pathlib import Path


# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config as cfg
import src.util as util
import src.hard_constraint as hc

cfg.load(ROOT / "config/templete.yaml")

print(f"device: {cfg.DEVICE}")

n = cfg.GRID_DENSITY


x, y = np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing="ij")
X =torch.tensor(np.column_stack([x.ravel(), y.ravel()]))


util.save_grid("./out/out_poisson_g.dat",X,hc.G(X))
