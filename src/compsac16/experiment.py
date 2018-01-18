## learner

import numpy as np
import csv,re,stringcase
import sys
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist


chou_data = {}
feature_names = 'feature_names'
target_names = 'target_names'
target = 'target'
data = 'data'

data_arr = np.array([], dtype=str)
target_arr = np.array([], dtype=str)


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

        print(header+'Surprising')
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
            output += row['Surprising'] in (None, '') and '' or row['Surprising']
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
    tokens = regexp_tokenize(t, pattern='[a-zA-Z_]+')
    processed_text = ''
    for w in tokens:
        if w not in stopWords:
            w = stemmer.stem(w)
            processed_text = processed_text+' '+stringcase.lowercase(w)
    return processed_text


def proc_sum_desc(file_name):
    with open('../data/'+file_name+'.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        print('issue_id,summary,description,summary_proc,description_proc,Surprising')
        for row in reader:
            output = str(row['issue_id'] in (None, '') and '' or row['issue_id']) + ','
            output += row['summary'] in (None,'') and '' or row['summary']+','
            output += row['summary'] in (None,'') and '' or pre_proc_text(row['summary'])+','
            output += row['description'] in (None,'') and '' or row['description']+','
            output += row['description'] in (None, '') and '' or pre_proc_text(row['description'])+','
            output += row['Surprising'] in (None, '') and '' or row['Surprising']
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

        target_column = 'Surprising'
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
        target_names_arr = np.array(['Surprise', 'NotSurprise'])
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
            chou_data[target][rw_counter] = row['Surprising']
            rw_counter += 1

        chou_data[data] = chou_data[data].astype(int)
        chou_data[target] = chou_data[target].astype(int)

    return chou_data





def doExperiment():
    ambari='ambari'
    camel='camel'
    derby= 'derby'
    wicket='wicket'
    file= ambari
    #pre_process(file)
    #vec_process(file)
    print(load_data(file))

    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.over_sampling import SMOTE

    X = chou_data[data]
    y = chou_data[target]

    # Oversampling
    rus = RandomUnderSampler()
    X_folds = np.array_split(X, 10)
    y_folds = np.array_split(y, 10)
    scores = list()
    for k in range(10):
        # We use 'list' to copy, in order to 'pop' later on
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)
        X_resampled, y_resampled = rus.fit_sample(X_train, y_train)

    print(scores)


doExperiment()