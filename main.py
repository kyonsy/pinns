import fcnn
import sampling
import loss
import torch
import config
import numpy as np
import torch.optim as optim

epoch:int=config.epoch
num_plot=config.num_plot
num_sample=config.num_sample

model =fcnn.FCNN()
optimizer=optim.Adam(model.parameters(),lr=1e-3)

for _ in range(epoch):
    X=sampling.sample_random_default()
    # X= torch.tensor([[x/num_sample] for x in range(num_sample)])
    X0=torch.zeros_like(X)
    
    X.requires_grad_(True)
    X0.requires_grad_(True)
    
    Y=model(X)
    Y0=model(X0)
    
    optimizer.zero_grad()
    l=loss.dumped_oscillation(X,Y,X0,Y0,0.0)
    l.backward()
    optimizer.step()

X_out= torch.tensor([[x/num_plot] for x in range(num_plot)])
Y_out=model(X_out)
XY_out=torch.cat([X_out,Y_out],dim=1)
np.savetxt("out.dat",XY_out.detach().numpy(),"%.6f")





