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

p_components = FreqDist()
n_components = FreqDist()

all_components = []
label_components = []


def labeling_components(file_name):
    csvfile = open(file_name + '_proc.csv', newline='', encoding="UTF-8")
    reader = csv.DictReader(csvfile)
    for row in reader:
        component_column = row['component'] in (None, '') and 'null' or row['component']
        label = row[intent] in (None, '') and '0' or row[intent]
        print(component_column, label)
        components = re.split("; ", component_column)
        counter = 0
        for component in components:
            all_components.append(component)
            label_components.append(int(label))
            if int(label) == 1:
                if component in p_components:
                    p_components[component] += 1
                else:
                    p_components[component] = 1
            else:
                if component in n_components:
                    n_components[component] += 1
                else:
                    n_components[component] = 1
            print(component, label)

    csvfile.close()
    print(len(all_components))
    print(Counter(label_components))

    print("-----------------POSITIVE--------------------")
    for x in p_components.most_common():
        print(x)
    print("-----------------NEGATIVE--------------------")
    for y in n_components.most_common():
        print(y)

    p_components_set = frozenset(p_components)
    n_components_set = frozenset(n_components)

    print(p_components_set)
    print(n_components_set)

    inter = len(p_components_set.intersection(n_components_set))
    union = len(p_components_set.union(n_components_set))

    print(inter,union,inter/union)
    return


subject = ambari
intent = Performance
labeling_components(subject)
