import torch
import itertools
from . import config as cfg

def sample_random_default()->torch.Tensor:
    return torch.rand(cfg.num_sample,cfg.in_features)

def sample_random(num_sample:int,in_features:int)->torch.Tensor:
    return torch.rand(num_sample,in_features)

def sample_grid(num_sample:int,dim:int)->torch.Tensor:
    axis = [x/num_sample for x in range(num_sample)] 
    return torch.tensor(list(itertools.product(axis,repeat=dim)))
    
"""
角が原点に接した単位正方形の外周からサンプルをとる
"""
def sample_square_boundry(num_sample:int) -> torch.Tensor:  
    bottom = torch.tensor([[x/num_sample,0] for x in range(num_sample)])
    right  = torch.tensor([[1,x/num_sample] for x in range(num_sample)])
    top    = torch.tensor([[x/num_sample,1] for x in range(num_sample)])
    left   = torch.tensor([[0,x/num_sample] for x in range(num_sample)])
    return torch.vstack([bottom,right,top,left])

if __name__=="__main__":
    # print(sample_random_default())
    print(sample_grid(2,2))
    print(sample_square_boundry(100))
