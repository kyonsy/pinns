import torch
import config

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

def dumped_oscillation(X:torch.Tensor,Y:torch.Tensor,X0:torch.Tensor,Y0:torch.Tensor,zeta:float)->torch.Tensor: 
    Y_t  = torch.autograd.grad(Y,X,torch.ones_like(X),create_graph=True)[0]
    Y_tt = torch.autograd.grad(Y_t,X,torch.ones_like(Y_t),create_graph=True)[0]
    Y_t0 = torch.autograd.grad(Y0,X0,torch.ones_like(X0),create_graph=True)[0]
    
    l_ode = ((Y_tt + 2*zeta*Y_t + Y)**2).mean()
    l_ic1 = ((Y0 - 1)**2).mean()
    l_ic2 = (Y_t0**2).mean()
    
    # print(zeta)
    # print(type(zeta))
    
    l= l_ode+l_ic1+l_ic2
    return l



if __name__=='__main__':
    import sampling
    import fcnn

    model= fcnn.FCNN()
    X=sampling.sample_random_default() * 20
    X0=torch.zeros_like(X)
    
    X.requires_grad_(True)
    X0.requires_grad_(True)
    
    Y=model(X)
    Y0=model(X0)
    
    l=dumped_oscillation(X,Y,X0,Y0,0.0)


    
    