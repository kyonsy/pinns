import torch
import config
in_features=config.in_features
num_sample=config.num_sample

def sample_random_default()->torch.Tensor:
    return torch.rand(num_sample,in_features)

def sample_random(num_sample:int,in_features:int)->torch.Tensor:
    return torch.rand(num_sample,in_features)

if __name__=="__main__":
    print(sample_random_default())
