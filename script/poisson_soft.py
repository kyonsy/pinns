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

# 入力するデータを選択
type = "test"

# コマンドライン引数がある場合は、そちらを type に代入
if len(sys.argv) > 1:
    type = sys.argv[1] 

# ソフト制約では境界条件を損失関数で課すので, 出力を変換しない素の FCNN を使う
cfg.load(ROOT / f"config/poisson_{type}.yaml")

print(f"device: {cfg.DEVICE}")
model=model.FCNN().to(cfg.DEVICE)
train.train_poisson_soft(model)
X_out= sampling.sample_grid(cfg.GRID_DENSITY,[1.0,1.0])
with torch.no_grad():
    Y_out=model(X_out)

util.save_grid(ROOT / f"out/out_poisson_{type}.dat",X_out,Y_out)
print(f"complete")
