from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import csv
import sys,re

file = 'ambari'
#sys.stdout = open("data/proc/"+file+"_sum_vec.csv","w")
stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()
words = []
with open("data/main/"+file+".csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        counter = 0
        # write header
        with open('data/word.txt', 'r') as txtfile:
            header = '';
            for line in txtfile:
                line = re.sub('\n', '', line)
                header += line + ','
        txtfile.close()
        print('issue_id,'+header+',Surprising, Dormant,Blocker, Security, Performance,Breakage')
        for row in reader:
            counter = counter + 1;
            summary = regexp_tokenize(row['summary'], pattern='\w+')
            wordsFiltered = []
            for w in summary:
                if w not in stopWords:
                    w = stemmer.stem(w)
                    wordsFiltered.append(w)
                if w not in words:
                    words.append(w);
            summary = wordsFiltered
            procSummary = '';
            for w in summary:
                procSummary+= ' '+w
            procSummary+= ''
            output = row['issue_id'] in (None,'') and '' or row['issue_id'] + ','
            output += row['summary'] in (None, '') and '' or row['summary']+","
            output += row['summary'] in (None, '') and '' or procSummary + ","
            output += row['description'] in (None, '') and '' or row['description']+","
            output += row['Surprising'] in (None, '') and '' or row['Surprising']+","
            output += row['Dormant'] in (None, '') and '' or row['Dormant']+","
            output += row['Blocker'] in (None, '') and '' or row['Blocker']+","
            output += row['Security'] in (None, '') and '' or row['Security']+","
            output += row['Performance'] in (None, '') and '' or row['Performance']+","
            output += row['Breakage'] in (None, '') and '' or row['Breakage']+","
            print(output)
#sys.stdout.close()
print(reader)
