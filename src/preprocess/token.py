from nltk.tokenize import RegexpTokenizer
import csv
with open('data/main/ambari.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        counter=counter+1;
        tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
        summary = tokenizer.tokenize(row['summary'])
        print(row['issue_id'],'{\'summary\':\'',row['summary'],summary,'\'}')
    print(counter)
print("Finished!")