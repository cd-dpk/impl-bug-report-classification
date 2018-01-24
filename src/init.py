from nltk.corpus import wordnet
word_to_test ='username'

if wordnet.synsets(word_to_test):
    print('Yes')
else:
    print('No')