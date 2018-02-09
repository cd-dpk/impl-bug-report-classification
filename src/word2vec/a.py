from gensim.models.word2vec import Word2Vec
import math
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('f.txt', binary=False)
vecs = word_vectors['host']
value = 0
for x in range(len(vecs)):
    # print(vecs[x] * vecs[x])
    value += vecs[x] * vecs[x]
    print(value)
print(value)

print(.38*.38)
print(math.sqrt(.144))


