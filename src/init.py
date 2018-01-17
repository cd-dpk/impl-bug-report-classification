from sklearn.datasets import load_iris
import sys
#sys.stdout= open('iris.txt','w')
#iris = load_iris()
#print(iris)
#sys.stdout.close()

import numpy as np
#[[1, 1], [2, 2], ]
a = np.empty([])
ar= np.array([3, 3])
a = np.insert(a,len(a),ar,axis=0)
ar= np.array([2, 2])
a = np.append(a,ar,axis=1)

print(a)
