from nltk.tokenize import regexp_tokenize
import csv
with open('data/main/ambari.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    counter = 0
    for row in reader:
        counter=counter+1;
        summary = regexp_tokenize(row['summary'],pattern='\w+')
        print(row['issue_id'],'{\'summary\':\'',row['summary'],summary,'\'}')
        print(counter)
print("Finished!")