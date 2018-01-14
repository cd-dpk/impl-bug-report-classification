from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
import csv
import sys,re

def pre_proc_text(t):
    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    summary = regexp_tokenize(t, pattern='\w+')
    proc_t= FreqDist()
    for w in summary:
        if w not in stopWords:
            w = stemmer.stem(w)
            proc_t[w.lower()] += 1
    return proc_t.most_common()


def proc(file):
    with open("data/main/" + file + ".csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        print('issue_id,summary,proc_summary,Surprising,Dormant,'
              'Blocker,Security,Performance,Breakage')
        for row in reader:
            terms = pre_proc_text(row['summary'])
            output = str(row['issue_id'] in (None, '') and '' or row['issue_id']) + ','
            output += row['summary'] in (None, '') and '' or row['summary'] + ","
            output += str(row['summary'] in (None, '') and '' or terms) + ","
            output += row['Surprising'] in (None, '') and '' or row['Surprising'] + ","
            output += row['Dormant'] in (None, '') and '' or row['Dormant'] + ","
            output += row['Blocker'] in (None, '') and '' or row['Blocker'] + ","
            output += row['Security'] in (None, '') and '' or row['Security'] + ","
            output += row['Performance'] in (None, '') and '' or row['Performance'] + ","
            output += row['Breakage'] in (None, '') and '' or row['Breakage'] + ","
            print(output)


file = 'ambari'
sys.stdout= open('temp.csv','w')
proc(file)
sys.stdout.close()