import math
import torch
import numpy as np

import fcnn as fcnn
import config as config
import sampling as sampling
import train as train

epoch:int=config.epoch
num_plot=config.num_plot
num_sample=config.num_sample
in_features=config.in_features

def main():  
    test2()

def test1():
    for i  in range(1):
        j=i+1
        scale = j * math.pi/4
    
        model =fcnn.FCNN()
        train.train_dump_sys(model)
        
        X_out= sampling.sample_grid(num_sample,in_features) * scale
        Y_out=model(X_out)
        XY_out=torch.cat([X_out,Y_out],dim=1)
        
        np.savetxt(f"out/out{j}.dat",XY_out.detach().numpy(),"%.6f")
        print(f"complete {j}")
        
def test2():
    model=fcnn.FCNN()
    train.train_poisson_eq(model)
    X_out= sampling.sample_grid(num_sample,in_features)
    Y_out=model(X_out)
    XY_out=torch.cat([X_out,Y_out],dim=1)
    
    np.savetxt(f"./out/out.dat",XY_out.detach().numpy(),"%.6f")
    print(f"complete") 
        
if __name__=="__main__":
    main()








