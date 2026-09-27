import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
(ROOT / "out").mkdir(exist_ok=True)  # clone直後は out/ が無いので作っておく

import torch
import math
import numpy as np
import src.fcnn as fcnn
import src.config as cfg
import src.sampling as sampling
import src.train as train

cfg.load(ROOT / "config/dump_sys.yaml")


for i  in range(1):
    j=i+1
    t_range = j * math.pi/4
    
    model =fcnn.FCNN()
    train.train_dump_sys(model,t_range)
    
    X_out= sampling.sample_grid(cfg.plot_density,[1.0])
    Y_out=model(X_out)
    XY_out=torch.cat([X_out,Y_out],dim=1)
    
    np.savetxt(ROOT / f"out/out{j}.dat",XY_out.detach().numpy(),"%.6f")
    print(f"complete {j}")
    









