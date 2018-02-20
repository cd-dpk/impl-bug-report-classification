## learnerimport csv,re,stringcase
import sys,csv,re,stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist


ambari = 'ambari'
camel = 'camel'
derby = 'derby'
wicket = 'wicket'

additionalStopWords = []
for x in range(26):
    additionalStopWords.append(chr(ord('a') + x))
vocabulary = FreqDist()

def freq_count(t):
    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    t = re.sub('[_]',' ', t)
    tokens = regexp_tokenize(t, pattern='[a-zA-Z]+')

    for w in tokens:
        w = stringcase.lowercase(w)
        if w not in stopWords and w not in additionalStopWords:
            if w in vocabulary:
                vocabulary[w] += 1
            else:
                vocabulary[w] = 1
    return vocabulary


def proc_src(file_name):
    with open('/media/geet/Files/IITDU/MSSE-03/SRC_P/'+file_name+'_term.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        print('term,freq')
        counter=0
        for row in reader:
            # if counter in (5901,7108,7373,7402,7460,7472,11755):
                # counter+=1
                # continue
            print(counter,row['proc'] in (None, '') and '' or row['proc'])
            freq_count(row['proc'] in (None, '') and '' or row['proc'])
            counter+=1

    n = int(len(vocabulary)*.1);
    for w in vocabulary.most_common(n):
        print(w)
    print(len(vocabulary))

    return

def pre_process(file_name):
#    sys.stdout= open(file_name+'_proc.csv','w')
    proc_src(file_name)
#   sys.stdout.close()
    return

'''Preprocess Ends Here'''

subject = sys.argv[1]
pre_process(subject)