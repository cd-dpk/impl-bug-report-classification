import numpy as np
from sklearn.naive_bayes import  MultinomialNB


data=np.array([np.array(['a',1,1],dtype=object),['a',0,1]])
target = np.array([0,1])
print(data)

estimator = MultinomialNB()
estimator.fit(data,target)

pre=estimator.predict(data)
print(pre)
from sklearn.datasets import load_diabetes
#print(load_diabetes())