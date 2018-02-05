import sys,csv,re,stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist

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

perf_keywords=["performance","slow","speed","latency","throughput",
      "cpu","disk","memory","usage","resource","calling",
      "times","infinite","loop"]
sec_keywords=["add","denial","service","XXE","remote","open","redirect","OSVDB","vuln","CVE","XSS","ReDoS",
          "NVD","malicious","frame","attack","exploit","directory","traversal","RCE","dos","XSRF",
          "clickjack","session","fixation","hijack","advisory","insecure","security","cross",
          "origin","unauthori[z|s]ed","authenticat(e|ion)","brute force","bypass","credential",
          "DoS","expos(e|ing)","hack","harden","injection","lockout","over flow","password","PoC",
          "proof","poison","privelage","(in)?secur(e|ity)","(de)?serializ", "spoof","traversal"]

def predict_keywords(t):
    # t = re.sub('[_]',' ', t)
    # t = re.sub('(?<=[A-Z])(?=[A-Z][a-z])|(?<=[^A-Z])(?=[A-Z])|(?<=[A-Za-z])(?=[^A-Za-z])',' ',t)
    tokens = regexp_tokenize(t, pattern='[a-zA-Z]+')
    sec_cl = 0
    perf_cl= 0
    for w in tokens:
        # print(w)
        for s in sec_keywords:
            s = "(?i)"+s
            if re.search(s,w):
                sec_cl=1
                break
        # print(sec_cl)
        for p in perf_keywords:
            p = "(?i)" + p
            if re.search(p,w):
                perf_cl=1
                break
        # print(perf_cl)
        if sec_cl ==1 and perf_cl == 1:
            break
    return (sec_cl,perf_cl)


# which single character is removed
additionalStopWords = []
for x in range(26):
    additionalStopWords.append(chr(ord('a') + x))


############### vocabulary of project#######################
vocabulary = FreqDist()

def freq_count(t):
    stopWords = set(stopwords.words('english'))
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
            print(counter,row['proc'] in (None, '') and '' or row['proc'])
            freq_count(row['proc'] in (None, '') and '' or row['proc'])
            counter+=1
    return

################# Ends ############


def pre_proc_sentence(t):
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
        ## when to use percentage of vocabulary
        ## n = int(len(vocabulary)*0.1)
        ## If src vocabulary is used,then
        # if w not in stopWords and w not in additionalStopWords and w not in vocabulary.most_common(n):
        ## otherwise
        if w not in stopWords and w not in additionalStopWords:
            w = stemmer.stem(w)
            processed_text = processed_text+' '+ w
    return processed_text


def proc_file(file_name):
    with open('../data/'+file_name+'.csv', newline='', encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        print('issue_id,reporter,component,keywords,summary_proc,description_proc,' + intent+",files")
        for row in reader:
            issue_id = str(row['issue_id'] in (None, '') and '' or row['issue_id'])
            reporter = row['reporter'] in (None,'') and 'null' or row['reporter']
            component = row['component'] in (None, '') and 'null' or row['component']
            summary_proc= (row['summary'] in (None,'') and '' or pre_proc_sentence(row['summary']))
            description_proc = (row['description'] in (None, '') and '' or pre_proc_sentence(row['description']))
            label = row[intent] in (None, '') and '0' or row[intent]
            sec , perf = predict_keywords((row['summary'] in (None,'') and''or'')+" "+(row['description'] in (None, '') and '' or ''))
            # sec, perf = predict_keywords((row['summary'] in (None, '') and '' or ''))
            files = row['files'] in (None, '') and '' or row['files']
            # print(issue_id+","+reporter+","+component+","+str(sec)+","+summary_proc+","+description_proc+","+label+","+files)
            print(issue_id+","+reporter+","+component+","+str(perf)+","+summary_proc+","+description_proc+","+label+","+files)
    return

def pre_process(file_name):
    sys.stdout= open(file_name+'_proc.csv','w')
    proc_file(file_name)
    sys.stdout.close()
    return

'''Preprocess Ends Here'''
subject = derby
intent = Performance
pre_process(subject)