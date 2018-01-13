from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import csv

stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()
words = []
with open('data/main/ambari.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    counter = 0
    print ('issue_id, type, status, resolution, component, priority, reporter, created, assigned, assignee, resolved, created, assigned, summary, description, affected_version, fixed_version, votes, watches, description_words, assingnee_count, comment_count, commenter, Surprising, Dormant, Blocker, Security, Performance, Breakage, commit_count, file_count, files')
    for row in reader:
        counter=counter+1;
        summary = regexp_tokenize(row['summary'],pattern='\w+')
        wordsFiltered = []
        for w in summary:
            if w not in stopWords:
                w = stemmer.stem(w)
                wordsFiltered.append(w)
                if w not in words:
                    words.append(w);
        summary = wordsFiltered
#        print((row['issue_id'] in (None,'')) and '' or row['issue_id'] + ',' + (row['issue_id'] in (None,'')) and '' or row['type'] + ',' + (row['issue_id'] in (None,'')) and '' or row['status'] + ',' + (row['issue_id'] in (None,'')) and '' or row['resolution'] + ',' + (row['issue_id'] in (None,'')) and '' or row[
#                'component'] + ',' + (row['issue_id'] in (None,'')) and '' or row['priority'] + ',' + (row['issue_id'] in (None,'')) and '' or row['reporter'] + ',' + (row['issue_id'] in (None,'')) and '' or row['created'] + ',' + (row['issue_id'] in (None,'')) and '' or row[
#                      'assigned'] + ',' + (row['issue_id'] in (None,'')) and '' or row['assignee'] + ',' + (row['issue_id'] in (None,'')) and '' or row['resolved'] + ',' + (row['issue_id'] in (None,'')) and '' or row['created'] + ',' + (row['issue_id'] in (None,'')) and '' or row[
#                      'assigned'] + ',' + (row['issue_id'] in (None,'')) and '' or row['summary'] + ',' + (row['issue_id'] in (None,'')) and '' or row['description'] + ',' + (row['issue_id'] in (None,'')) and '' or row[
#                      'affected_version'] + ',' + (row['issue_id'] in (None,'')) and '' or row['fixed_version'] + ',' + (row['issue_id'] in (None,'')) and '' or row['votes'] + ',' + (row['issue_id'] in (None,'')) and '' or row[
#                     'watches'] + ',' + (row['issue_id'] in (None,'')) and '' or row['description_words'] + ',' + (row['issue_id'] in (None,'')) and '' or row['assingnee_count'] + ',' + (row['issue_id'] in (None,'')) and '' or row[
#                      'comment_count'] + ',' + (row['issue_id'] in (None,'')) and '' or row['commenter'] + ',' + (row['issue_id'] in (None,'')) and '' or row['Surprising'] + ',' + (row['issue_id'] in (None,'')) and '' or row['Dormant'] + ',' +
#              (row['issue_id'] in (None, '')) and '' or row['Blocker'] + ',' + (row['issue_id'] in (None,'')) and '' or row['Security'] + ',' + (row['issue_id'] in (None,'')) and '' or row['Performance'] + ',' + (row['issue_id'] in (None,'')) and '' or row['Breakage'] + ',' + (row['issue_id'] in (None,'')) and '' or row[
#                      'commit_count'] + ',' + (row['issue_id'] in (None,'')) and '' or row['file_count'] + ',' + (row['issue_id'] in (None,'')) and '' or row['files'])
#     print(counter)
print("Finished!")
counter = 0
for w in words:
    counter = counter+1
print(counter)