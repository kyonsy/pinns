import math
import torch

"""
減衰振動 y'' + 2*zeta*y' + y = 0, y(0)=1, y'(0)=0 の解析解
zeta<1: 不足減衰, zeta=1: 臨界減衰, zeta>1: 過減衰
"""
def dumped_oscillation(X:torch.Tensor,zeta:float) -> torch.Tensor:
    if math.isclose(zeta,1.0):
        return (1 + X)*torch.exp(-X)
    elif zeta < 1.0:
        omega = (1.0 - zeta**2)**0.5
        return torch.exp(-zeta*X)*(torch.cos(omega*X) + (zeta/omega)*torch.sin(omega*X))
    else:
        omega = (zeta**2 - 1.0)**0.5
        r1 = -zeta + omega
        r2 = -zeta - omega
        return (-r2*torch.exp(r1*X) + r1*torch.exp(r2*X))/(r1 - r2)

if __name__=='__main__':
    from . import config as cfg
    import numpy as np

    num_plot=cfg.num_plot

    Z =[0.0,0.5,1.0,1.5,2.0,2.5]

    for zeta in Z:
        X=torch.tensor([[x/num_plot] for x in range(num_plot)])*20
        Y=dumped_oscillation(X,zeta)
        XY=torch.cat([X,Y],dim=1)
        np.savetxt(f"out/out{zeta}.txt",XY.detach().numpy(),"%.6f")
