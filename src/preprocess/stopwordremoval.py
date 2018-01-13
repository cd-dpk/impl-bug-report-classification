from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
import csv
def splitCamelCase( str):
 	   return str.replace(
 	      ("%s|%s|%s",
 	         "(?<=[A-Z])(?=[A-Z][a-z])",
 	         "(?<=[^A-Z])(?=[A-Z])",
 	         "(?<=[A-Za-z])(?=[^A-Za-z])"
 	      ),
 	      " "
 	   )

stopWords = set(stopwords.words('english'))
words = []
with open('data/main/ambari.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    counter = 0
    for row in reader:
        counter=counter+1;
        summary = regexp_tokenize(row['summary'],pattern='\w+')
        wordsFiltered = []
        for w in summary:
            if w not in stopWords:
                wordsFiltered.append(w)
                if w not in words:
                    words.append(w);
        summary = wordsFiltered
        print(row['issue_id'],'{\'summary\':\'',row['summary'],summary,'\'}')
        print(counter)
print("Finished!")
counter = 0
for w in words:
    counter = counter+1
print(counter)