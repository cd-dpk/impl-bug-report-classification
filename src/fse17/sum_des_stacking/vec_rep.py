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



def get_all_components(file):
    csvfile = open(file+'_proc.csv', newline='')
    reader = csv.DictReader(csvfile)
    component_list = []
    for row in reader:
        component_column = row['component'] in (None,'') and 'null' or row['component']
        candidate_components = regexp_tokenize(component_column, pattern='[a-zA-Z-]+')

        for component in candidate_components:
            if component not in component_list:
                component_list.append(component)

    csvfile.close()
    return component_list

def get_all_reporters(file):
    csvfile = open(file+'_proc.csv', newline='')
    reader = csv.DictReader(csvfile)
    reporter_list = []
    for row in reader:
        reporter = row['reporter'] in (None,'') and 'null' or row['reporter']
##        print(reporter)
        if reporter not in reporter_list:
            reporter_list.append(reporter)

    csvfile.close()
    return reporter_list


def get_all_terms(file):
    csvfile = open(file+'_proc.csv', newline='')
    reader = csv.DictReader(csvfile)
    word_list =[]
    word_df =[]
    for row in reader:
        text = row['summary_proc'] in (None,'') and '' or row['summary_proc']+" "+row['description_proc'] in (None,'') and '' or row['description_proc']
        dic = term_count(text)
        for term in dic:
            if term not in word_list:
                word_list.append(term)
                word_df.append(1)
            else:
                for x in range (len(word_list)):
                    if word_list[x] == term:
                        word_df[x] += 1

    csvfile.close()
    return (word_list,word_df)



def proc_sum_desc_vec(file_name):
    #print("Hello")
    word_list, word_df = get_all_terms(file_name)
    for x in range(len(word_df)):
        word_df[x] = float(word_df[x])
        word_df[x] = math.log((len(word_df)/word_df[x]),10)

    ## include header
    header =''
    for word in word_list:
        header +=word+','

    print(header + intent)

    csvfile = open(file_name+'_proc.csv', newline='')
    reader = csv.DictReader(csvfile)

    for row in reader:
        output = ''
        text = row['summary_proc']+" "+row['description_proc']
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
                #weight = terms[index][1] * word_df[index]
                #weight = round(weight,2)
                rw += str(weight)+','
            else:
                rw += '0,'
        output += rw
        output += row[intent] in (None, '') and '0' or row[intent]
        print(output)
    csvfile.close()
    return

def vec_process(file_name):
    sys.stdout= open(file_name+'_vec.csv','w')
    proc_sum_desc_vec(file_name)
    sys.stdout.close()
    return

'''Vector Representation Ends here'''

subject=sys.argv[1]
intent=sys.argv[2]
vec_process(subject)