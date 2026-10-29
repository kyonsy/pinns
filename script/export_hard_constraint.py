import sys
import numpy as np
import torch
from pathlib import Path


# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config
import src.sampling as sampling
import src.util as util
import src.hard_constraint as hc

cfg = config.load(ROOT / "config/templete.yaml")

print(f"device: {config.DEVICE}")

n = cfg.divisions()    # 分割数. 1辺の点数は n+1


# 数値解・PINN と同じ出力点
X = torch.tensor(sampling.output_grid(n), device=config.DEVICE)


util.save_grid(ROOT / "out/g.dat",X,hc.G(X,n))
util.save_grid(ROOT / "out/gl.dat",X,hc.G(X,n)*hc.L(X,n))

# 中心付近の断面 (y=b). C は格子間隔 h の幅で変化するので, 格子より細かく 401 点とる
# 列: x, C, G, L, G*L
P_k = hc.param(n)[0]
a, b = P_k[0,0].item(), P_k[3,0].item()
h = 1/n
xs = torch.linspace(b - 4*h, a + 4*h, 401, dtype=torch.float64, device=config.DEVICE)
Xs = torch.column_stack([xs, torch.full_like(xs, b)])
section = torch.column_stack([xs[:,None], hc.C(Xs,n), hc.G(Xs,n), hc.L(Xs,n), hc.G(Xs,n)*hc.L(Xs,n)])
np.savetxt(ROOT / "out/hc_section.dat", section.cpu().numpy(), fmt="%.8f", header="x C G L GL (y=b)")

# 中心付近の C を 2 次元で (3D 図用). [b-4h, a+4h]^2 を 121x121 点でとる
cs = torch.linspace(b - 4*h, a + 4*h, 121, dtype=torch.float64, device=config.DEVICE)
Xc = torch.cartesian_prod(cs, cs)    # x が外側, y が内側 (sample_grid と同じ並び)
util.save_grid(ROOT / "out/hc_C_center.dat", Xc, hc.C(Xc,n))
