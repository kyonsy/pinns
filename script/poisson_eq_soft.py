import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

import src.fcnn as fcnn
import src.config as cfg
import src.sampling as sampling
import src.train as train

# ソフト制約では境界条件を損失関数で課すので, 出力を変換しない素の FCNN を使う
cfg.load(ROOT / "config/poisson_eq_soft.yaml")

print(f"device: {cfg.device}")
model=fcnn.FCNN().to(cfg.device)
train.train_poisson_eq_soft(model)
X_out= sampling.sample_grid(cfg.plot_density,[1.0,1.0])
with torch.no_grad():
    Y_out=model(X_out)
XY_out=torch.cat([X_out,Y_out],dim=1)

np.savetxt(ROOT / "out/out_soft.dat",XY_out.cpu().numpy(),"%.6f")
print(f"complete")
