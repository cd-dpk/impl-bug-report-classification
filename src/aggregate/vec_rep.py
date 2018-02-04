## learner
import math
import csv,re,stringcase
import sys
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from collections import Counter

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
    word_list =[]
    word_df =[]
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
    word_list, word_df = get_all_terms(file_name)

    header_str  = ''
    header_str += 'reporter,'
    header_str += 'component,'
    header_str += 'keywords,'
    header_words = ''
    for word in word_list:
        header_words +=word+','

    print(header_str + header_words + intent)

    csvfile = open(file_name+'_proc.csv', newline='')
    reader = csv.DictReader(csvfile)

    for row in reader:
        output = ''
        output += row['reporter']+","
        output += row['component']+","
        output += row['keywords']+","

        text = row['summary_proc']
        dic = term_count(text)
        terms = dic.most_common()
        rw=''
        for x in range(len(word_list)):
            counter = 0
            index = -1
            for t in terms:
                if word_list[x] in (terms[counter]) :
                    index = counter
                    break
            if index != -1:
                weight = terms[index][1]
                rw += str(weight)+','
            else:
                rw += '0,'
        output += rw
        label = row[intent] in (None, '') and 'No' or row[intent]
        if label.startswith('1'):
            output += 'Yes'
        elif label.startswith('0'):
            output += 'No'

        print(output)
    csvfile.close()
    return

def vec_process(file_name):
    sys.stdout= open(file_name+'_vec.csv','w')
    proc_textual_info(file_name)
    sys.stdout.close()
    return

'''Vector Representation Ends here'''

subject=ambari
intent=Security
vec_process(subject)