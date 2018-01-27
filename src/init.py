import sys
print(str(sys.argv))

for x in range(3):
    print(x)


x = [2,3,4]

if 3 not in x and 2 not in x:
    print('No')
else:
    print('Yes')


from collections import Counter

y = [0,1,0,0]
c = Counter(y)
keys = c.keys()
for i in keys:
    print(c.__getitem__(i))
