import re, sklearn
print("Hello, World!")
from sklearn.feature_selection import SelectFdr, SelectKBest
from sklearn.feature_selection import chi2
X = [[1, 2], [2, 3], [4, 5]]
y = [0, 1, 0]

print(X)
print(y)
ch2 = chi2(X,y)
print(chi2(X,y))
selector = SelectKBest(chi2, k=1)
selector.fit(X, y)
print(selector.transform(X))
