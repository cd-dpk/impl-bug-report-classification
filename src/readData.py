import nltk
from nltk.corpus import stopwords
import csv
stopWords = set(stopwords.words('english'))
with open('data/main/ambari.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    counter = 0
    for row in reader:
        counter=counter+1;
        summary = nltk.word_tokenize(row['summary'],language='english',preserve_line='')
        wordsFiltered = []
        for w in summary:
            if w not in stopWords:
                wordsFiltered.append(w)
        summary = wordsFiltered

        print(row['issue_id'],'{\'summary\':\'',row['summary'],summary,'\'}')
    print(counter)
