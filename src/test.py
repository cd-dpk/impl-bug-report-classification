import numpy as np
a = np.array([1,2,3], dtype=int)
b = np.array([3,4,5], dtype=int)
agg = a+b
print(list(frozenset(agg)))
