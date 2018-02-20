## learner
import math
import csv, re, stringcase
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

p_files_list = FreqDist()
n_files_list = FreqDist()

all_files = []
label_files = []


def label_src_files(file_name):
    csvfile = open(file_name + '_proc.csv', newline='', encoding="UTF-8")
    reader = csv.DictReader(csvfile)
    for row in reader:
        file_column = row['files'] in (None, '') and 'No File' or row['files']
        label = row["target_"+intent] in (None, '') and '0' or row["target_"+intent]
        print(file_column, label)
        files = re.split(";", file_column)
        counter = 0
        for file in files:
            # if re.search("^camel-core/", file) and re.search(".java$", file):
            if re.search(".java$",file):
                all_files.append(file)
                label_files.append(int(label))
                if int(label) == 1:
                    if file in p_files_list:
                        p_files_list[file] += 1
                    else:
                        p_files_list[file] = 1
                else:
                    if file in n_files_list:
                        n_files_list[file] += 1
                    else:
                        n_files_list[file] = 1
                # print(file, label)
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
    print("-----------------POSITIVE--------------------")
    for x in p_files_list.most_common(50):
        print(x)
    print("-----------------NEGATIVE--------------------")
    for y in n_files_list.most_common(50):
        print(y)

    p_files_set = frozenset(p_files_list)
    n_files_set = frozenset(n_files_list)

    print(p_files_set)
    print(n_files_set)

    inter = len(p_files_set.intersection(n_files_set))
    union = len(p_files_set.union(n_files_set))

    print(inter, union, inter/union)

    return

subject = camel
intent = Security
label_src_files(subject)