## learnerimport csv,re,stringcase
import sys,csv,re,stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ambari = 'ambari'
camel = 'camel'
derby = 'derby'
wicket = 'wicket'
all = 'all'
subject =''
Surprising = 'Surprising'
Security = 'Security'
Performance = 'Performance'
intent = ''

additionalStopWords = []
for x in range(26):
    additionalStopWords.append(chr(ord('a') + x))

def pre_proc_text(t):
    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    t = re.sub('[_]',' ', t)
    ## camel case and Pascal Case splitted
    t = re.sub('(?<=[A-Z])(?=[A-Z][a-z])|(?<=[^A-Z])(?=[A-Z])|(?<=[A-Za-z])(?=[^A-Za-z])',' ',t)
    tokens = regexp_tokenize(t, pattern='[a-zA-Z]+')
    #print('Tokens:',tokens)

    processed_text = ''
    for w in tokens:
        w = stringcase.lowercase(w)
        if w not in stopWords and w not in additionalStopWords:
            w = stemmer.stem(w)
            processed_text = processed_text+' '+ w
    #print('Processed Text:',processed_text)
    return processed_text


def proc_sum_desc(file_name):
    with open('/media/geet/Files/IITDU/MSSE-03/implementation/src/data/'+file_name+'.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        print('issue_id,summary,summary_proc,description,description_proc,' + intent)
        for row in reader:
            output = str(row['issue_id'] in (None, '') and '' or row['issue_id']) + ','
            output += row['summary'] in (None,'') and '' or row['summary']+','
            output += row['summary'] in (None,'') and '' or pre_proc_text(row['summary'])+','
            output += (row['description'] in (None,'') and '' or row['description'])+','
            output += (row['description'] in (None, '') and '' or pre_proc_text(row['description']))+','
            output += row[intent] in (None, '') and '0' or row[intent]
            print(output)
    return

def pre_process(file_name):
    sys.stdout= open(file_name+'_proc.csv','w')
    proc_sum_desc(file_name)
    sys.stdout.close()
    return

'''Preprocess Ends Here'''

subject = sys.argv[1]
intent= sys.argv[2]
pre_process(subject)