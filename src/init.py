import numpy as np

a = np.empty(10)
print(a)
x = 0
for i in range(10):
    print(i)
    print(i)
    print('Hello')
    print(i)
    a[x] = i
    x +=1

print(a)