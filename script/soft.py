import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

import src.fcnn as fcnn
import src.config as config
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
cfg = config.load(ROOT / f"config/{type}.yaml")

print(f"device: {config.DEVICE}")
if cfg.HARD_BOUNDARY:
    model=fcnn.SoftCenterFCNN(cfg).to(config.DEVICE)
    train.train_soft_center(model,cfg)
else:
    model=fcnn.FCNN(cfg).to(config.DEVICE)
    train.train_soft(model,cfg)
# 出力は数値解と同じ点 (sampling.output_grid) でとる. 座標は float64 のまま保存し, モデルには float32 で入れる
X_out= torch.tensor(sampling.output_grid(cfg.divisions()),device=config.DEVICE)
with torch.no_grad():
    Y_out=model(X_out.float())

util.save_grid(ROOT / f"out/{type}.dat",X_out,Y_out)
print(f"complete")
