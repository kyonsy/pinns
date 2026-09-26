import torch
import config

in_features=config.in_features
num_sample=config.num_sample
num_plot=config.num_plot

def sample_random_default()->torch.Tensor:
    return torch.rand(num_sample,in_features)

def sample_random(num_sample:int,in_features:int)->torch.Tensor:
    return torch.rand(num_sample,in_features)

"""
電磁ポテンシャルの境界条件用
角が原点に接した1×1の正方形の外周に関するサンプルをとる
"""
def sample_rectangle_boundry(n:int) -> torch.Tensor:
    X=torch.rand(n,3)
    k=n//4
    X[:k,0]=0.
    X[k:2*k,0]=1.
    X[2*k:3*k,1]=0.
    X[3*k:1]=1.
    X[:,3]=0.
    return X

if __name__=="__main__":
    # print(sample_random_default())
    print(sample_boundry(100,3))
