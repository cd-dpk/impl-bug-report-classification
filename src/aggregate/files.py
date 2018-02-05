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
subject = ''
Surprising = 'Surprising'
Security = 'Security'
Performance = 'Performance'
intent = ''

files_list = FreqDist()

all_files = []
label_files = []

def label_src_files(file_name):
    csvfile = open(file_name+'_proc.csv', newline='', encoding="utf-8")
    reader = csv.DictReader(csvfile)
    for row in reader:
        file_column = row['files'] in (None, '')and 'No File' or row['files']
        label = row[intent] in (None, '')and '0' or row[intent]
        print(file_column,label)
        files = re.split(";",file_column)
        counter = 0
        for file in files:
            # if re.search("^camel-core/",file) and re.search(".java$", file):
            if re.search(".java$",file):
                all_files.append(file)
                label_files.append(int(label))
                if int(label) == 1:
                    if file in files_list:
                        files_list[file] += 1
                    else:
                        files_list[file] = 1

            print(file, label)
            counter += 1
        print(counter)
    csvfile.close()

    print(len(all_files))
    print(Counter(label_files))
    '''
    for x in range(len(all_files)):
        if label_files[x] == 1:
            print(all_files[x])
    '''
    for x in files_list.most_common():
        print(x)
    return


subject = derby
intent = Performance
label_src_files(subject)