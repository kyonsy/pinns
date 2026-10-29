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

# 入力するデータを選択
type = "soft1"

# コマンドライン引数がある場合は、そちらを type に代入
if len(sys.argv) > 1:
    type = sys.argv[1] 

# ソフト制約では境界条件を損失関数で課すので, 出力を変換しない素の FCNN を使う
# yaml で HARD_BOUNDARY: true なら, 外周の φ=0 だけ出力の変換で厳密に課し, 損失は方程式と中心の4点だけにする
cfg.load(ROOT / f"config/{type}.yaml")

print(f"device: {cfg.DEVICE}")
if cfg.HARD_BOUNDARY:
    model=fcnn.SoftCenterFCNN().to(cfg.DEVICE)
    train.train_soft_center(model)
else:
    model=fcnn.FCNN().to(cfg.DEVICE)
    train.train_soft(model)
# 出力は x=1, y=1 の外周も含めて正方形全体 [0,1]x[0,1] でとる
X_out= sampling.sample_grid_closed(cfg.divisions(),[1.0,1.0])
with torch.no_grad():
    Y_out=model(X_out)

util.save_grid(ROOT / f"out/{type}.dat",X_out,Y_out)
print(f"complete")
