#減衰振動に関するPINNについて、ネットワークの解と解析解の間の誤差と、減衰比との関係を調べる
#学習区間は[0,2π]で固定

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

zetas=[(x+1) * 0.1 for x in range(20)]
trange = 2*math.pi

for i,zeta in enumerate(zetas,start=1):
    model =fcnn.FCNN()
    train.train_dump_sys(model,trange,zeta)
    
    # PINNの解と解析解を同じ点で計算し, 誤差と並べて保存する
    X_out   = sampling.sample_grid(cfg.plot_density,[trange])
    with torch.no_grad():
        Y_pinn  = model(X_out)
    Y_exact = analysis.dumped_oscillation(X_out,zeta)
    abs_err = (Y_pinn - Y_exact).abs()
    data    = torch.cat([X_out,Y_pinn,Y_exact,abs_err],dim=1)
    
    np.savetxt(ROOT / f"out/zeta{zeta}.dat",data.numpy(),"%.6e",header="t pinn exact abs_err")
    print(f"complete zeta={zeta}  [{i}/{len(zetas)}] {i/len(zetas):.0%}")
    

