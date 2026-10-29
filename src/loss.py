import torch

def loss_hard(X:torch.Tensor,Y:torch.Tensor) -> torch.Tensor:
    """
    ポアソンの方程式に関するPINNの損失関数
    入力が2つ(x,y). 出力が一つ(φ)
    境界条件(外周で φ=0, 中心の4点で φ=1)はモデル側(PoissonFCNN)で固定している
    """
    dY = torch.autograd.grad(Y,X,torch.ones_like(Y),create_graph=True)[0]
    phi_x,phi_y = dY[:,0:1],dY[:,1:2]
    
    phi_xx = torch.autograd.grad(phi_x,X,torch.ones_like(phi_x),create_graph=True)[0][:,0:1]
    phi_yy = torch.autograd.grad(phi_y,X,torch.ones_like(phi_y),create_graph=True)[0][:,1:2]
    
    L_ode  = ((phi_xx + phi_yy)**2).mean()
    return L_ode

def loss_soft(X:torch.Tensor,Y:torch.Tensor,Y0:torch.Tensor,Y1:torch.Tensor,w:torch.Tensor) -> torch.Tensor:
    """
    ポアソンの方程式に関するPINNの損失関数(ソフト制約)
    入力が2つ(x,y). 出力が一つ(φ)
    Y0: 外周での出力 (φ=0 にしたい)
    Y1: 中心の4点での出力 (φ=1 にしたい)
    w : 重み (Config.LOSS_WEIGHTS) = [方程式, 外周, 中心] の順
    """
    L_ode = loss_hard(X,Y)
    L_bc0 = (Y0**2).mean()
    L_bc1 = ((Y1 - 1)**2).mean()

    L = w[0] * L_ode + w[1] * L_bc0 + w[2] * L_bc1
    return L
