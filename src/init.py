from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re,stringcase

def pre_proc_text(t):
    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    t = re.sub('[_]',' ', t)
    ## camel case and Pascal Case splitted
    t = re.sub('(?<=[A-Z])(?=[A-Z][a-z])|(?<=[^A-Z])(?=[A-Z])|(?<=[A-Za-z])(?=[^A-Za-z])',' ',t)
    tokens = regexp_tokenize(t, pattern='[a-zA-Z_]+')
    processed_text = ''
    for w in tokens:
        if w not in stopWords:
            w = stemmer.stem(w)
            processed_text = processed_text+' '+stringcase.lowercase(w)
    return processed_text


text="it yarn-site having:'yarn.nodemanager.local-dirs' : '/grid/0/hadoop/yarn /grid/1/hadoop/yarn"
print(pre_proc_text(text))