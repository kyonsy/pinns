import torch
import source.config as config

loss_weights=config.loss_weights
in_features= config.in_features

"""y=x^2に関するPINNsの損失関数"""
def parabola(X:torch.Tensor,Y:torch.Tensor)->torch.Tensor:
    Z=((Y-X**2)**2)
    l=Z.mean()
    return l


"""
鉛直投げ上げ問題に関するPINNの損失関数
最大の高さが, 落下する時間が1となるように無次元化してある
入力と出力が1次元で動作
"""
def vertical_throw(X:torch.Tensor,Y:torch.Tensor,X0:torch.Tensor,Y0:torch.Tensor)->torch.Tensor:  
    Y_t  = torch.autograd.grad(Y,X,torch.ones_like(X),create_graph=True)[0]
    Y_tt = torch.autograd.grad(Y_t,X,torch.ones_like(Y_t),create_graph=True)[0]
    Y_t0 = torch.autograd.grad(Y0,X0,torch.ones_like(X0),create_graph=True)[0]
    
    l_ode = ((Y_tt+8)**2).mean()
    l_ic1 = (Y0**2).mean()
    l_ic2 = ((Y_t0-4)**2).mean()
    
    l= l_ode+l_ic1+l_ic2
    
    return l

def loss_dump_sys(X:torch.Tensor,Y:torch.Tensor,X0:torch.Tensor,Y0:torch.Tensor,zeta:float)->torch.Tensor: 
    Y_t  = torch.autograd.grad(Y,X,torch.ones_like(X),create_graph=True)[0]
    Y_tt = torch.autograd.grad(Y_t,X,torch.ones_like(Y_t),create_graph=True)[0]
    Y_t0 = torch.autograd.grad(Y0,X0,torch.ones_like(X0),create_graph=True)[0]
    
    # 単振動を再現できない原因を解明するため、一時的に角速度ωを10に設定
    L_ode = ((Y_tt + 2*zeta*Y_t + Y)**2).mean()
    L_ic1 = ((Y0 - 1)**2).mean()
    L_ic2 = (Y_t0**2).mean()
    
    # print(zeta)
    # print(type(zeta))
    
    L= L_ode+L_ic1+L_ic2
    return L

"""
ポアソンの方程式に関するPINNの損失関数
入力が2つ(x,y). 出力が一つ(φ)
Y0,Y1は境界条件
"""
def loss_poisson_sys(X:torch.Tensor,Y:torch.Tensor,Y0:torch.Tensor,Y1:torch.Tensor) -> torch.Tensor:
    dY     = torch.autograd.grad(Y,X,torch.ones_like(Y),create_graph=True)[0]
    phi_x,phi_y = dY[:,0:1],dY[:,1:2]
    
    phi_xx = torch.autograd.grad(phi_x,X,torch.ones_like(phi_x),create_graph=True)[0][:,0:1]
    phi_yy = torch.autograd.grad(phi_y,X,torch.ones_like(phi_y),create_graph=True)[0][:,1:2]
    
    L_ode  = ((phi_xx + phi_yy)**2).mean()
    L_bc1  = (Y0**2).mean()
    L_bc2  = ((Y1-1)**2).mean()
    
    L = L_ode + L_bc1 + L_bc2
    return L
    
if __name__=='__main__':
    import source.sampling as sampling
    import source.fcnn as fcnn

    model= fcnn.FCNN()
    X=sampling.sample_grid(100,2)
    # print(f"Sample in grid {X}\n")
    # print(f"Shape: {X.shape}")
    # K=sampling.sample_random(100,2)
    # print(f"Sample in random {K}")
    # print(f"Shape: {K.shape}")
    
    X0=sampling.sample_square_boundry(100)
    X1=torch.tensor([[0.5,0.5]])
    
    X.requires_grad_(True)
    X0.requires_grad_(True)
    X1.requires_grad_(True)
    
    Y=model(X)
    Y0=model(X0)
    Y1=model(X1)
    
    l=loss_poisson_sys(X,Y,Y0,Y1)
    
    


    
    