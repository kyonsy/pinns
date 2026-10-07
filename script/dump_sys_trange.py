#減衰振動に関するPINNについて、ネットワークの解と解析解の間の誤差と、学習区間との関係を調べる
#減衰比は0で固定(つまり単振動について学習を行う)

import sys
from pathlib import Path

# プロジェクトルート (script/ の1つ上). どこから実行しても src を import できるようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import math
import numpy as np
import src.fcnn as fcnn
import src.config as cfg
import src.sampling as sampling
import src.train as train
import src.analysis as analysis

cfg.load(ROOT / "config/dump_sys.yaml")
print(f"device: {cfg.device}")


scale = 20
for i  in range(scale):
    j=i+1
    trange = j * math.pi/2
    
    model =fcnn.FCNN().to(cfg.device)
    train.train_dump_sys(model,trange,0.0)
    
    # PINNの解と解析解を同じ点で計算し, 誤差と並べて保存する
    X_out   = sampling.sample_grid(cfg.plot_density,[trange])
    with torch.no_grad():
        Y_pinn  = model(X_out)
    Y_exact = analysis.dumped_oscillation(X_out,0.0)
    abs_err = (Y_pinn - Y_exact).abs()
    data    = torch.cat([X_out,Y_pinn,Y_exact,abs_err],dim=1)
    
    np.savetxt(ROOT / f"out/out{j}.dat",data.cpu().numpy(),"%.6e",header="t pinn exact abs_err")
    print(f"complete[{j}/scale] {j/scale:.0%}")
    









