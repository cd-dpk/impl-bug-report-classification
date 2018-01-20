import numpy as np
from sklearn.naive_bayes import  MultinomialNB

from collections import Counter

def add(x:int,y:int):
    return (x,y)

a,b = add('2',1)

print(a)
print(b)

a = np.array([[10,10],[20,10]])
print(a[:][:][0])

ar = [0,0,0,1]
a_dic = Counter(ar)
print(a_dic)