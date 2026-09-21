import matplotlib.pyplot as plt
import numpy as np

d=np.loadtxt("out.txt")
plt.plot(d[:,0],d[:,1],"o",ms=3)

t=np.linspace(0,1,200)
plt.xlabel("t")
plt.ylabel("y")
plt.show()