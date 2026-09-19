import torch
import config
loss_weights=config.loss_weights

def physics_loss(X:torch.Tensor,Y:torch.Tensor)->torch.Tensor:
    Z=((Y-X**2)**2)
    l=Z.mean()
    return l

if __name__=='__main__':
    import sampling
    C=sampling.sample_random(100,1)
    l=physics_loss(C,C)
    print(l)
    
    