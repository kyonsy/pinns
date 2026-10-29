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

# 出力ファイル名 -> モデルのクラス
models = {
    "hard1" : fcnn.PoissonFCNN1,
    "hard2" : fcnn.PoissonFCNN2,
}

# 入力するデータを選択
type = "hard1"

# コマンドライン引数がある場合は、そちらを type に代入
if len(sys.argv) > 1:
    type = sys.argv[1]

cfg.load(ROOT / "config/hard.yaml")

print(f"device: {cfg.DEVICE}")

model=models[type]().to(cfg.DEVICE)

train.train_hard(model)
# 出力は数値解と同じ点 (sampling.output_grid) でとる. 座標は float64 のまま保存し, モデルには float32 で入れる
X_out= torch.tensor(sampling.output_grid(cfg.divisions()),device=cfg.DEVICE)
with torch.no_grad():
    Y_out=model(X_out.float())

util.save_grid(ROOT / f"out/{type}.dat",X_out,Y_out)
print(f"complete")
