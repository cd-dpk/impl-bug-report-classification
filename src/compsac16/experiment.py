## learner

import numpy as np
import csv,re,stringcase
import sys
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist

intent='Performance'
chou_data = {}
feature_names = 'feature_names'
target_names = 'target_names'
target = 'target'
data = 'data'

data_arr = np.array([], dtype=str)
target_arr = np.array([], dtype=str)

additionalStopWords = []
for x in range(26):
    additionalStopWords.append(chr(ord('a') + x))


def term_count(t):
    summary = regexp_tokenize(t, pattern='[a-zA-Z]+')
    proc_t= FreqDist()
    for w in summary:
        proc_t[w.lower()] += 1
    return proc_t

def get_all_terms(file):
    with open(file+'_proc.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        word_list =[]
        for row in reader:
            text = row['summary_proc'] in (None,'') and '' or row['summary_proc']+" "+row['description_proc'] in (None,'') and '' or row['description_proc']
            dic = term_count(text)
            terms = dic.most_common()
            for term in dic:
                if term not in word_list:
                    word_list.append(term)
        return word_list


def proc_sum_desc_vec(file_name):
    word_list= get_all_terms(file_name)
    #print(word_list)
    with open(file_name+'_proc.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        ## include header
        header =''
        for word in word_list:
            header+=word+','

        print(header+intent)
        for row in reader:
            text = row['summary']+" "+row['description']
            dic = term_count(text)
            terms = dic.most_common()
            output = ''
            rw=''
            for word in word_list:
                counter = 0
                index = -1
                for t in terms:
                    if word in (terms[counter]) :
                        index = counter
                        break
                if index != -1:
                    rw += str(terms[index][1])+','
                else:
                    rw += '0,'
            output += rw
            output += row[intent] in (None, '') and '0' or row[intent]
            print(output)

def vec_process(file_name):
    sys.stdout= open(file_name+'_vec.csv','w')
    proc_sum_desc_vec(file_name)
    sys.stdout.close()
    return


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
    with open('../data/'+file_name+'.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        print('issue_id,summary,description,summary_proc,description_proc,'+intent)
        for row in reader:
            output = str(row['issue_id'] in (None, '') and '' or row['issue_id']) + ','
            output += row['summary'] in (None,'') and '' or row['summary']+','
            output += row['summary'] in (None,'') and '' or pre_proc_text(row['summary'])+','
            output += row['description'] in (None,'') and '' or row['description']+','
            output += row['description'] in (None, '') and '' or pre_proc_text(row['description'])+','
            output += row[intent] in (None, '') and '0' or row[intent]
            print(output)
    return

def pre_process(file_name):
    sys.stdout= open(file_name+'_proc.csv','w')
    proc_sum_desc(file_name)
    sys.stdout.close()
    return


def setFeatureNamesAndRows(file_name):
    with open(file_name+'_vec.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        ## set features names
        # print(reader.fieldnames)
        feature_names_arr = np.array(reader.fieldnames)

        # print(feature_names_arr)

        target_column = intent
        if feature_names_arr[len(feature_names_arr) - 1] == target_column:
            feature_names_arr = np.delete(feature_names_arr, len(feature_names_arr) - 1, axis=0)

        row_count = 0
        for row in reader:
            row_count += 1

        chou_data[feature_names] = feature_names_arr
        # print(chou_data[feature_names])

        ##initialize empty np_array(data,target) with shape of row_count and feature_count

        # print(str(row_count)+','+str(len(feature_names_arr)))
        data_arr = np.empty([row_count, len(feature_names_arr)], dtype=str)
        target_arr = np.empty([row_count], dtype=str)
        chou_data[data] = data_arr
        chou_data[target] = target_arr
        return


def load_data(file):
    setFeatureNamesAndRows(file)
    # print(chou_data[feature_names])
    with open(file+'_vec.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # set target names
        target_names_arr = np.array([intent, 'Not'+intent])
        chou_data[target_names] = target_names_arr

        rw_counter = 0
        # set data and target
        for row in reader:
            f_counter = 0
            data_arr_row = np.empty([len(chou_data[feature_names])], dtype=str)
            features = chou_data[feature_names]
            for x in features:
                if row[x] not in (None, ''):
                    data_arr_row[f_counter] = row[x]
                f_counter += 1

            chou_data[data][rw_counter] = data_arr_row
            chou_data[target][rw_counter] = row[intent]
            rw_counter += 1

        chou_data[data] = chou_data[data].astype(int)
        print(chou_data[target])
        chou_data[target] = chou_data[target].astype(int)

    return chou_data




def doExperiment():
    ambari='ambari'
    camel='camel'
    derby= 'derby'
    wicket ='wicket'
    file = wicket
    #pre_process(file)
    #exit()
    #vec_process(file)
    #exit()
    print(load_data(file))

    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.over_sampling import SMOTE
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import precision_recall_fscore_support
    X = chou_data[data]
    y = chou_data[target]

    X_folds = np.array_split(X, 10)
    y_folds = np.array_split(y, 10)

    precision_arr = np.empty([3,10],dtype=float)
    recall_arr = np.empty([3,10],dtype=float)
    '''fm_arr = np.empty([3,10],dtype=float)'''

    for k in range(10):
        # We use 'list' to copy, in order to 'pop' later on
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)

        estimator = MultinomialNB();

        ## Under Sampling ##
        rus = RandomUnderSampler()
        X_rus, y_rus = rus.fit_sample(X_train, y_train)
        estimator.fit(X_rus,y_rus)
        y_predict = estimator.predict(X_test)
        t_p = 0.0
        t_n = 0.0
        f_p = 0.0
        f_n = 0.0
        for i in range(len(y_predict)):
            if y_test[i] == 1:
               if y_predict[i] == 1:
                   t_p +=1.0
               else:
                   f_n +=1
            if y_test[i] == 0:
                if y_predict[i] == 1:
                    f_p +=1
                else:
                    t_n +=1


        print(t_p,f_p,t_n,f_n)

        precision = (t_p)/(t_p+f_p)
        recall = (t_p)/(t_p+f_n)
        '''fm = (1.0/precision)+(1.0/recall)
        fm = 1.0/fm
        '''
        precision_arr[0][k] = precision
        recall_arr[0][k] = recall
        '''fm_arr[0][k] = fm'''

        ## Over Sampling ##
        ros = RandomOverSampler()
        X_ros, y_ros = ros.fit_sample(X_train, y_train)
        estimator.fit(X_ros, y_ros)
        y_predict = estimator.predict(X_test)
        t_p = 0.0
        t_n = 0.0
        f_p = 0.0
        f_n = 0.0
        for i in range(len(y_predict)):
            if y_test[i] == 1:
                if y_predict[i] == 1:
                    t_p += 1.0
                else:
                    f_n += 1
            if y_test[i] == 0:
                if y_predict[i] == 1:
                    f_p += 1
                else:
                    t_n += 1

        precision = (t_p) / (t_p + f_p)
        recall = (t_p) / (t_p + f_n)
        '''fm = (1.0 / precision) + (1.0 / recall)
        fm = 1.0 / fm'''
        precision_arr[1][k] = precision
        recall_arr[1][k] = recall
        '''fm_arr[1][k] = fm'''

        ## SMOTE ##
        sm = SMOTE()
        X_sm, y_sm = sm.fit_sample(X_train, y_train)
        estimator.fit(X_sm, y_sm)
        y_predict = estimator.predict(X_test)
        t_p = 0.0
        t_n = 0.0
        f_p = 0.0
        f_n = 0.0
        for i in range(len(y_predict)):
            if y_test[i] == 1:
                if y_predict[i] == 1:
                    t_p += 1.0
                else:
                    f_n += 1
            if y_test[i] == 0:
                if y_predict[i] == 1:
                    f_p += 1
                else:
                    t_n += 1

        precision = (t_p) / (t_p + f_p)
        recall = (t_p) / (t_p + f_n)
        '''fm = (1.0 / precision) + (1.0 / recall)
        fm = 1.0 / fm'''
        precision_arr[2][k] = precision
        recall_arr[2][k] = recall
        '''fm_arr[2][k] = fm'''


    print(round(np.array(precision_arr[0]).mean(),4), round(np.array(precision_arr[1]).mean(),4),round(np.array(precision_arr[2]).mean(),4))
    print(round(np.array(recall_arr[0]).mean(),4), round(np.array(recall_arr[1]).mean(),4),round(np.array(recall_arr[2]).mean(),4))
    '''print(round(np.array(fm_arr[0]).mean(),4), round(np.array(fm_arr[1]).mean(),4),round(np.array(fm_arr[2]).mean(),4))'''


doExperiment()