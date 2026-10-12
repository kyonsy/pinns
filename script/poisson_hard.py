import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

import src.fcnn as fcnn
import src.config as cfg
import src.sampling as sampling
import src.train as train
import src.util as util

cfg.load(ROOT / "config/poisson_hard.yaml")

print(f"device: {cfg.device}")
model=fcnn.PoissonFCNN().to(cfg.device)
train.train_poisson_hard(model)
X_out= sampling.sample_grid(cfg.plot_density,[1.0,1.0])
with torch.no_grad():
    Y_out=model(X_out)

util.save_grid(ROOT / f"out/out_hard.dat",X_out,Y_out)
print(f"complete") 