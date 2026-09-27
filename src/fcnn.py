import torch
import torch.nn as nn
import config as config

in_features: int = config.in_features
out_features: int = config.out_features
num_layers: int = config.num_layers
width: int = config.width
Activation = config.Activation


class FCNN(nn.Module):
    first: nn.Linear
    hidden: nn.ModuleList
    final: nn.Linear
    activations: nn.ModuleDict

    def __init__(self) -> None:
        super().__init__()
        self.first=nn.Linear(in_features,width)
        self.hidden = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(num_layers)]
        )
        self.final = nn.Linear(width, out_features)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        linear: nn.Module
        Y=self.first(X)
        for linear in self.hidden:
            Y = linear(Y)
            Y = Activation(Y)
        Y = self.final(Y)
        return Y

if __name__=='__main__':
    import fcnn as fcnn
    import sampling as sampling
    
    model: FCNN = FCNN()
    X=sampling.sample_random_default() * 20
    X0=torch.zeros_like(X)
    
    X.requires_grad_(True)
    X0.requires_grad_(True)
    
    Y=model(X)
    Y0=model(X0)
    
    optimizer.zero_grad()
    l=loss.dumped_oscillation(X,Y,X0,Y0,0)