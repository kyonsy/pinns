import fcnn
import sampling
import loss
import torch
import config
import numpy as np
import torch.optim as optim

epoch:int=config.epoch
num_plot=config.num_plot

model =fcnn.FCNN()
# X=sampling.sample_random(config.num_sample,config.in_features)
# Y=model(X)
# print(Y)
# l=loss.physics_loss(Y,X)
# print(l)
optimizer=optim.SGD(model.parameters(),lr=1e-2)
# optimizer.zero_grad()
# print(model.final.weight.grad)
# print("\n")
# print(list(model.final.parameters()))
# print("\n")
# l.backward()
# print(list(model.final.weight.grad))
# print("\n")
# optimizer.step()
# print(list(model.final.parameters()))
# print(sampling.sample_random_default())

for _ in range(epoch):
    X=sampling.sample_random_default()
    X.requires_grad=False
    Y=model(X)
    optimizer.zero_grad()
    l=loss.physics_loss(X,Y)
    l.backward()
    optimizer.step()

X_out= torch.tensor([[x/num_plot] for x in range(num_plot)])
Y_out=model(X_out)
XY_out=torch.cat([X_out,Y_out],dim=1)
np.savetxt("out.txt",XY_out.detach().numpy(),"%.6f")
print(X_out)
print(Y_out)
print(XY_out)

    




