dictionary = {'a': 1}
dictionary.__setitem__('b',2)
print(dictionary['b'])

from nltk import FreqDist
a = FreqDist()
a['1'] = 1
a['2'] = 2
print(a.most_common())