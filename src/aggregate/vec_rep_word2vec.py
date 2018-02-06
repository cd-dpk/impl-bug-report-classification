## learner
import math
import csv,re,stringcase
import sys
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from collections import Counter
from gensim.models.word2vec import Word2Vec

ambari = 'ambari'
camel = 'camel'
derby = 'derby'
wicket = 'wicket'
all = 'all'
subject =''
Surprising = 'Surprising'
Security = 'Security'
Performance = 'Performance'
intent = ''




def term_count(t):
    summary = regexp_tokenize(t, pattern='[a-zA-Z]+')
    proc_t= FreqDist()
    for w in summary:
        proc_t[w.lower()] += 1
    return proc_t


def get_all_terms(file):
    csvfile = open(file+'_proc.csv', newline='')
    reader = csv.DictReader(csvfile)
    word_list = []
    word_df = []
    for row in reader:
        # text = row['summary_proc'] in (None,'') and '' or row['summary_proc']+" "+row['description_proc'] in (None,'') and '' or row['description_proc']
        text = row['summary_proc'] in (None,'') and '' or row['summary_proc']
        terms = term_count(text)
        for term in terms:
            if term not in word_list:
                word_list.append(term)
                word_df.append(1)
            else:
                for x in range (len(word_list)):
                    if word_list[x] == term:
                       word_df[x] += 1

    csvfile.close()
    return (word_list,word_df)

def proc_textual_info(file_name):
    csvfile = open(file_name+'_proc.csv', newline='')
    reader = csv.DictReader(csvfile)
    sentences =[]
    for row in reader:
        text = row['summary_proc']
        line_sentence = []
        text = regexp_tokenize(text,"[a-zA-Z]+")
        for t in text:
            line_sentence.append(t)

        sentences.append(line_sentence)

    for sentence in sentences:
        print(sentence)

    model = Word2Vec(sentences,size=100,window=5,min_count=1)
    model.wv.save_word2vec_format("f.txt", binary=False)

    csvfile.close()
    return

def vec_process(file_name):
    #sys.stdout= open(file_name+'_vec.csv','w')
    proc_textual_info(file_name)
    #sys.stdout.close()
    return

'''Vector Representation Ends here'''

subject = ambari
intent = Security
vec_process(subject)