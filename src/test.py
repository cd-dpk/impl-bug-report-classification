import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
import re

stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()
t = 'Add additional transition states'
## camel case and Pascal Case splitted
s2 =' '+re.sub('(?<=[A-Z])(?=[A-Z][a-z])|(?<=[^A-Z])(?=[A-Z])|(?<=[A-Za-z])(?=[^A-Za-z])',' ',t)
print(s2)
summary = regexp_tokenize(s2, pattern='[a-zA-Z_]+')

print(summary)
