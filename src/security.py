from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
import csv
import sys,re

def getAllWords(file):
    with open(file, newline='') as csvfile:
        stopWords = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        reader = csv.DictReader(csvfile)
        word_list = FreqDist()
        for row in reader:
#            if row['Security'] == '0':
#                continue
            if ['Performance'] == '0':
                continue
            text = row['summary']
#            print(text)
            summary = regexp_tokenize(text, pattern='[a-zA-Z_]+')
            for w in summary:
                if w not in stopWords:
                    #w = stemmer.stem(w)
                    word_list[w.lower()] += 1
        return word_list

file = 'compsac_16.csv'
sys.stdout= open('compsac_16_perf.txt','w')
words = getAllWords(file)
print(words.most_common())
sys.stdout.close()
