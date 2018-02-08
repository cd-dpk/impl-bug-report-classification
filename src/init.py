import numpy as np

a = np.array([1,2],dtype=int)
b = np.array([3,4],dtype=int)
print(np.concatenate((a,b),axis=0))
print(np.concatenate((a,b),axis=1))