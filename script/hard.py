import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

import src.model as model
import src.config as cfg
import src.sampling as sampling
import src.train as train
import src.util as util

cfg.load(ROOT / "config/hard.yaml")

print(f"device: {cfg.DEVICE}")

# クラスを変えればモデルが切り替わる
model=model.PoissonFCNN().to(cfg.DEVICE)

train.train_hard(model)
X_out= sampling.sample_grid(cfg.GRID_DENSITY,[1.0,1.0])
with torch.no_grad():
    Y_out=model(X_out)

util.save_grid(ROOT / f"out/hard.dat",X_out,Y_out)
print(f"complete") 