from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import csv
import sys
file = 'wicket'
sys.stdout = open("data/proc/"+file+".csv","w")
stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()
words = []
with open("data/main/"+file+".csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        counter = 0
        print('issue_id, type, status, resolution, component, priority, reporter, created, '
              'assigned, assignee, resolved, created, assigned,summary,proc_summary,description, '
              'affected_version, fixed_version, votes, watches, description_words, '
              'assingnee_count, comment_count,  Surprising, Dormant, '
              'Blocker, Security, Performance, Breakage, commit_count, file_count, files')
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
            output += row['type'] in (None,'') and '' or row['type']+","
            output += row['status'] in (None, '') and '' or row['status']+","
            output += row['resolution'] in (None, '') and '' or row['resolution']+","
            output += row['component'] in (None, '') and '' or row['component']+","
            output += row['priority'] in (None, '') and '' or row['priority']+","
            output += row['reporter'] in (None, '') and '' or row['reporter']+","
            output += row['created'] in (None, '') and '' or row['created']+","
            output += row['assigned'] in (None, '') and '' or row['assigned']+","
            output += row['assignee'] in (None, '') and '' or row['assignee']+","
            output += row['resolved'] in (None, '') and '' or row['resolved']+","
            output += row['created'] in (None, '') and '' or row['created']+","
            output += row['assigned'] in (None, '') and '' or row['assigned']+","
            output += row['summary'] in (None, '') and '' or row['summary']+","
            output += row['summary'] in (None, '') and '' or procSummary + ","
            output += row['description'] in (None, '') and '' or row['description']+","
            output += row['affected_version'] in (None, '') and '' or row['affected_version']+","
            output += row['fixed_version'] in (None, '') and '' or row['fixed_version']+","
            output += row['votes'] in (None, '') and '' or row['votes']+","
            output += row['watches'] in (None, '') and '' or row['watches']+","
            output += row['description_words'] in (None, '') and '' or row['description_words']+","
            output += row['assingnee_count'] in (None, '') and '' or row['assingnee_count']+","
            output += row['comment_count'] in (None, '') and '' or row['comment_count']+","
            output += row['Surprising'] in (None, '') and '' or row['Surprising']+","
            output += row['Dormant'] in (None, '') and '' or row['Dormant']+","
            output += row['Blocker'] in (None, '') and '' or row['Blocker']+","
            output += row['Security'] in (None, '') and '' or row['Security']+","
            output += row['Performance'] in (None, '') and '' or row['Performance']+","
            output += row['Breakage'] in (None, '') and '' or row['Breakage']+","
            output += row['commit_count'] in (None, '') and '' or row['commit_count']+","
            output += row['file_count'] in (None, '') and '' or row['file_count']+","
            output += row['files'] in (None, '') and '' or row['files']+","
            print(output)
        print(counter)
#print("Finished!")
#counter = 0
#for w in words:
#    counter = counter+1
#print(counter)