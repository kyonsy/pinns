import sys
import numpy as np
import torch
from pathlib import Path


# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config as cfg
import src.util as util
import src.hard_constraint as hc

cfg.load(ROOT / "config/templete.yaml")

print(f"device: {cfg.DEVICE}")

n = cfg.divisions()    # 分割数. 1辺の点数は n+1


# x=1, y=1 の外周も含めて正方形全体 [0,1]x[0,1] の (n+1)^2 点 (sampling.sample_grid_closed と同じ)
x, y = np.meshgrid(np.arange(n+1) / n, np.arange(n+1) / n, indexing="ij")
X =torch.tensor(np.column_stack([x.ravel(), y.ravel()]))


util.save_grid("./out/g.dat",X,hc.G(X))
util.save_grid("./out/gl.dat",X,hc.G(X)*hc.L(X))

# 中心付近の断面 (y=b). C は格子間隔 h の幅で変化するので, 格子より細かく 401 点とる
# 列: x, C, G, L, G*L
P_k = hc.param()[0]
a, b = P_k[0,0].item(), P_k[3,0].item()
h = 1/n
xs = torch.linspace(b - 4*h, a + 4*h, 401, dtype=torch.float64)
Xs = torch.column_stack([xs, torch.full_like(xs, b)])
section = torch.column_stack([xs[:,None], hc.C(Xs), hc.G(Xs), hc.L(Xs), hc.G(Xs)*hc.L(Xs)])
np.savetxt(ROOT / "out/hc_section.dat", section.numpy(), fmt="%.8f", header="x C G L GL (y=b)")

# 中心付近の C を 2 次元で (3D 図用). [b-4h, a+4h]^2 を 121x121 点でとる
cs = torch.linspace(b - 4*h, a + 4*h, 121, dtype=torch.float64)
Xc = torch.cartesian_prod(cs, cs)    # x が外側, y が内側 (sample_grid と同じ並び)
util.save_grid(ROOT / "out/hc_C_center.dat", Xc, hc.C(Xc))
