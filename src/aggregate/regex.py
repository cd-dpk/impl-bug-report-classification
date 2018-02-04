## learnerimport csv,re,stringcase
import sys,csv,re,stringcase, numpy as np
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

perf=["performance","slow","speed","latency","throughput",
      "cpu","disk","memory","usage","resource","calling",
      "times","infinite","loop"]
sec=["add","denial","service","XXE","remote","open","redirect","OSVDB","vuln","CVE","XSS","ReDoS",
          "NVD","malicious","frame","attack","exploit","directory","traversal","RCE","dos","XSRF",
          "clickjack","session","fixation","hijack","advisory","insecure","security","cross",
          "origin","unauthori[z|s]ed","authenticat(e|ion)","brute force","bypass","credential",
          "DoS","expos(e|ing)","hack","harden","injection","lockout","over flow","password","PoC",
          "proof","poison","privelage","(in)?secur(e|ity)","(de)?serializ", "spoof","traversal"]

additionalStopWords = []
for x in range(26):
    additionalStopWords.append(chr(ord('a') + x))

def predict_class(t):
    # t = re.sub('[_]',' ', t)
    # t = re.sub('(?<=[A-Z])(?=[A-Z][a-z])|(?<=[^A-Z])(?=[A-Z])|(?<=[A-Za-z])(?=[^A-Za-z])',' ',t)
    tokens = regexp_tokenize(t, pattern='[a-zA-Z]+')
    sec_cl = 0
    perf_cl= 0
    for w in tokens:
        # print(w)
        for s in sec:
            s = "(?i)"+s
            if re.search(s,w):
                sec_cl=1
                break
        # print(sec_cl)
        for p in perf:
            p = "(?i)" + p
            if re.search(p,w):
                perf_cl=1
                break
        # print(perf_cl)
        if sec_cl ==1 and perf_cl == 1:
            break

    return (sec_cl,perf_cl)

def confusion_matrix(y_test,y_predict):
    t_p = 0.0
    t_n = 0.0
    f_p = 0.0
    f_n = 0.0
    for i in range(len(y_predict)):
        if y_test[i] == 1:
            if y_predict[i] == 1:
                t_p += 1.0
            else:
                f_n += 1.0
            continue
        if y_test[i] == 0:
            if y_predict[i] == 1:
                f_p += 1.0
            else:
                t_n += 1.0

    return {'t_p':t_p,'f_p':f_p,'t_n':t_n,'f_n':f_n}

def calc_pre_rec(result_dic:dict):
    t_p = result_dic['t_p']
    t_n = result_dic['t_n']
    f_p = result_dic['f_p']
    f_n = result_dic['f_n']

    if (t_p+f_p) != 0 and (t_p+f_n) != 0:
        pre = t_p/(t_p+f_p)
        rec = t_p/(t_p+f_n)
        return (pre,rec)
    else:
        return (0.0,0.0)

def calc_accuracy(result_dic:dict):
    t_p = result_dic['t_p']
    t_n = result_dic['t_n']
    f_p = result_dic['f_p']
    f_n = result_dic['f_n']

    return (t_p+t_n)/(t_p+t_n+f_p+f_n)

def proc_sum_desc(file_name):
    y_target=np.empty([4000,2],dtype=int)
    y_predict = np.empty([4000,2],dtype=int)

    with open('/media/geet/Files/IITDU/MSSE-03/implementation/src/data/'+subject+'.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        counter = 0
        for row in reader:
            summary= row['summary'] in (None,'') and '' or row['summary']
            description= (row['description'] in (None,'') and '' or row['description'])
            print(summary)
            print(description)
            y_target[counter][0] = int(row[Security] in (None, '') and '0' or row[Security])
            y_target[counter][1] = int(row[Performance] in (None, '') and '0' or row[Performance])
            print(y_target[counter][0],y_target[counter][1])
            y_predict[counter][0],y_predict[counter][1] = predict_class(summary+" "+description)
            counter+=1

        print(y_target)
        print(y_predict)

        y_target_1,y_predict_1=np.empty(4000,dtype=int),np.empty(4000,dtype=int)
        y_target_2,y_predict_2 = np.empty(4000,dtype=int),np.empty(4000,dtype=int)
        for x in range(len(y_target_1)):
            y_target_1[x] = y_target[x][0]
            y_target_2[x] = y_target[x][1]

            y_predict_1[x] = y_predict[x][0]
            y_predict_2[x] = y_predict[x][1]

    print(confusion_matrix(y_target_1,y_predict_1))
    print(confusion_matrix(y_target_2,y_predict_2))
    print(calc_pre_rec(confusion_matrix(y_target_1,y_predict_1)))
    print(calc_pre_rec(confusion_matrix(y_target_2,y_predict_2)))
    print(calc_accuracy(confusion_matrix(y_target_1,y_predict_1)))
    print(calc_accuracy(confusion_matrix(y_target_2,y_predict_2)))
    return

def pre_process(file_name):
    proc_sum_desc(file_name)
    return

'''Preprocess Ends Here'''
subject=wicket
pre_process(subject)