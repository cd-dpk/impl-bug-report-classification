import numpy as np
import sys,csv
from collections import Counter

ambari = 'ambari'
camel = 'camel'
derby = 'derby'
wicket = 'wicket'
all = 'all'
Surprising = 'Surprising'
Security = 'Security'
Performance = 'Performance'

subject =''
intent = ''
chou_data = {}
feature_names = 'feature_names'
target_names = 'target_names'
target = 'target'
data = 'data'

data_arr = np.array([], dtype=object)
target_arr = np.array([], dtype=object)

def setFeatureNamesAndRows(file_name):
    with open(file_name+'_vec.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        feature_names_arr = np.array(reader.fieldnames)
        target_column = intent
        if feature_names_arr[len(feature_names_arr) - 1] == target_column:
            feature_names_arr = np.delete(feature_names_arr, len(feature_names_arr) - 1, axis=0)

        row_count = 0
        for row in reader:
            row_count += 1

        chou_data[feature_names] = feature_names_arr
        data_arr = np.empty([row_count, len(feature_names_arr)], dtype=object)
        target_arr = np.empty([row_count], dtype=object)
        chou_data[data] = data_arr
        chou_data[target] = target_arr
        return


def load_data(file):
    setFeatureNamesAndRows(file)
    with open(file+'_vec.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # set target names
        target_names_arr = np.array([intent, 'Not'+intent])
        chou_data[target_names] = target_names_arr

        rw_counter = 0
        # set data and target
        for row in reader:
            f_counter = 0
            data_arr_row = np.empty([len(chou_data[feature_names])], dtype=object)
            features = chou_data[feature_names]
            for x in features:
                if x in ('reporter','component'):
                    data_arr_row[f_counter] = row[x]
                elif row[x] not in (None, ''):
                    data_arr_row[f_counter] = int(row[x],10)

                f_counter += 1

            chou_data[data][rw_counter] = data_arr_row
            chou_data[target][rw_counter] = int(row[intent],10)
            rw_counter += 1

    return chou_data

def under_sampling(X, y):
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()
        X_s, y_s = rus.fit_sample(X, y)
        return (X_s, y_s)


def over_sampling(X, y):
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    X_s, y_s = ros.fit_sample(X, y)
    return (X_s, y_s)


def smote(X, y):
    from imblearn.over_sampling import SMOTE
    sm = SMOTE()
    X_s, y_s = sm.fit_sample(X, y)
    return (X_s, y_s)


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
                f_n += 1
            continue
        if y_test[i] == 0:
            if y_predict[i] == 1:
                f_p += 1
            else:
                t_n += 1

    return {'t_p':t_p,'f_p':f_p,'t_n':t_n,'f_n':f_n}

def calc_tuple(result_dic:dict):
    return (result_dic['t_p'],result_dic['t_n'],result_dic['f_p'],result_dic['f_n'])

def calc_acc_pre_rec(result_dic:dict):
    t_p = result_dic['t_p']
    t_n = result_dic['t_n']
    f_p = result_dic['f_p']
    f_n = result_dic['f_n']

    if (t_p+f_p) != 0 and (t_p+f_n) != 0:
        pre = t_p/(t_p+f_p)
        rec = t_p/(t_p+f_n)
        acc = (t_p+t_n)/(t_p+f_p+t_n+f_n)
        return (acc,pre,rec)
    else:
        return (0.0,0.0,0.0)

def calc_test(result_dic:dict):
    t_p = result_dic['t_p']
    t_n = result_dic['t_n']
    f_p = result_dic['f_p']
    f_n = result_dic['f_n']
    return (t_p,t_n)

def doExperiment(file):
    print(load_data(file))
    print(Counter(chou_data[target]))

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest
    X = chou_data[data]
    y = chou_data[target]

    #X = SelectKBest(chi2(X,y),k=200)

    X_folds = np.array_split(X, 10)
    y_folds = np.array_split(y, 10)

    t_p = 0.0
    f_p = 0.0
    t_n = 0.0
    f_n = 0.0

    h = MultinomialNB()
    for k in range(10):
        # We use 'list' to copy, in order to 'pop' later on
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)

        X_s, y_s = under_sampling(X_train, y_train)
        #X_s, y_s = over_sampling(X_train, y_train)
        #X_s, y_s = smote(X_train, y_train)

        h.fit(X_s,y_s)
        y_predict = h.predict(X_test)
        print(confusion_matrix(y_test, y_predict))
        temp_tp, temp_tn,temp_fp,temp_fn = calc_tuple(confusion_matrix(y_test, y_predict))
        t_p += temp_tp
        t_n += temp_tn
        f_p += temp_fp
        f_n += temp_fn

    print(t_p,t_n,f_p,f_n)
    print(calc_acc_pre_rec({'t_p':t_p,'f_p':f_p,'t_n':t_n,'f_n':f_n}))
''' Experiment Ends here '''

subject= ambari
intent= Security
doExperiment(subject)
