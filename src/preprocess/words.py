from nltk.tokenize import regexp_tokenize
from nltk.stem import PorterStemmer
import csv

file = 'ambari'
stemmer = PorterStemmer()
words = []
with open("data/proc/" + file + ".csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    counter = 0
    for row in reader:
        counter = counter + 1;
        proc_summary = row['proc_summary'] and '' or row['proc_summary']
        if proc_summary not in (None, ''):
            proc_summary = regexp_tokenize(proc_summary, pattern='\w+')
            for w in proc_summary:
                if w not in words:
                    words.append(w)
print(len(words))
print(words)

with open('data/word.txt', 'w') as txtfile:
    for w in words:
        txtfile.write(w + "\n")


def readWords(str):
    import re
    with open('str', 'r') as txtfile:
        for line in txtfile:
            line = re.sub('\n','',line)
        words.setdefault(line,0)
    print(words)
    return words