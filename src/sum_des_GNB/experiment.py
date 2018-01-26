import numpy as np
import sys,csv
from collections import Counter

subject =''
intent = ''
chou_data = {}
feature_names = 'feature_names'
target_names = 'target_names'
target = 'target'
data = 'data'

data_arr = np.array([], dtype=str)
target_arr = np.array([], dtype=str)

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



def ensemble_confusion_matrix(y_test,y1_predict,y2_predict,y3_predict):
    t_p = 0.0
    t_n = 0.0
    f_p = 0.0
    f_n = 0.0

    for x in range(len(y_test)):
        y_predict = 0;
        pos =0
        neg =0
        if y1_predict[x] == 1:
            pos+=1
        else:
            neg+=1

        if y2_predict[x] == 1:
            pos+=1
        else:
            neg+=1

        if y3_predict[x] == 1:
            pos+=1
        else:
            neg+=1

        if pos >= neg:
            y_predict=1

        if y_test[x] == 1:
            if y_predict == 1:
                t_p += 1.0
            else:
                f_n += 1
        if y_test[x] == 0:
            if y_predict == 1:
                f_p += 1
            else:
                t_n += 1

    return {'t_p':t_p,'f_p':f_p,'t_n':t_n,'f_n':f_n}

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



def doExperiment(file):
    print(load_data(file))
    print(Counter(chou_data[target]))
    #exit()
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import svm
    X = chou_data[data]
    y = chou_data[target]

    X_folds = np.array_split(X, 10)
    y_folds = np.array_split(y, 10)

    pre = np.empty(3, dtype=float)
    rec = np.empty(3, dtype=float)
    fm = np.empty(3,dtype=float)
    pre[0] = 0.0
    pre[1] = 0.0
    pre[2] = 0.0
    rec[0] = 0.0
    rec[1] = 0.0
    rec[2] = 0.0
    fm[0] = 0.0
    fm[1] = 0.0
    fm[2] = 0.0

    for k in range(10):
        # We use 'list' to copy, in order to 'pop' later on
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)

        h1 = MultinomialNB();
        h2 = KNeighborsClassifier(5,weights='distance')
        h3 = svm.SVC()
        ## under_sampling ##
        X_s, y_s = under_sampling(X_train, y_train)

        h1.fit(X_s, y_s)
        y1_predict = h1.predict(X_test)

        '''
        h2.fit(X_s, y_s)
        y2_predict = h2.predict(X_test)

        h3.fit(X_s, y_s)
        y3_predict = h3.predict(X_test)
        '''

        print(confusion_matrix(y_test, y1_predict))
        temp_pre, temp_rec = calc_pre_rec(confusion_matrix(y_test, y1_predict))
        print(temp_pre,temp_rec)

        '''
        print(ensemble_confusion_matrix(y_test, y1_predict,y2_predict,y3_predict))
        temp_pre, temp_rec = calc_pre_rec(ensemble_confusion_matrix(y_test, y1_predict,y2_predict,y3_predict))
        '''
        pre[0] = pre[0] + temp_pre
        rec[0] = rec[0] + temp_rec


        ## over_sampling ##
        X_s, y_s = over_sampling(X_train, y_train)

        h1.fit(X_s, y_s)
        y1_predict = h1.predict(X_test)
        '''
        h2.fit(X_s, y_s)
        y2_predict = h2.predict(X_test)

        h3.fit(X_s, y_s)
        y3_predict = h3.predict(X_test)
        '''

        print(confusion_matrix(y_test, y1_predict))
        temp_pre, temp_rec = calc_pre_rec(confusion_matrix(y_test, y1_predict))
        print(temp_pre, temp_rec)
        '''
        print(ensemble_confusion_matrix(y_test, y1_predict,y2_predict,y3_predict))
        temp_pre, temp_rec = calc_pre_rec(ensemble_confusion_matrix(y_test, y1_predict,y2_predict,y3_predict))
        '''
        pre[1] = pre[1] + temp_pre
        rec[1] = rec[1] + temp_rec


        ## smote ##
        X_s, y_s = smote(X_train, y_train)

        h1.fit(X_s, y_s)
        y1_predict = h1.predict(X_test)

        '''
        h2.fit(X_s, y_s)
        y2_predict = h2.predict(X_test)

        h3.fit(X_s, y_s)
        y3_predict = h3.predict(X_test)
        '''

        print(confusion_matrix(y_test, y1_predict))
        temp_pre, temp_rec = calc_pre_rec(confusion_matrix(y_test, y1_predict))
        print(temp_pre, temp_rec)
        '''
        print(ensemble_confusion_matrix(y_test, y1_predict,y2_predict,y3_predict))
        temp_pre, temp_rec = calc_pre_rec(ensemble_confusion_matrix(y_test, y1_predict,y2_predict,y3_predict))
        '''
        pre[2] = pre[2] + temp_pre
        rec[2] = rec[2] + temp_rec


    pre[0] = pre[0] / 10.0
    pre[1] = pre[1] / 10.0
    pre[2] = pre[2] / 10.0

    rec[0] = rec[0] / 10.0
    rec[1] = rec[1] / 10.0
    rec[2] = rec[2] / 10.0

    fm[0] = 2.0/((1.0/pre[0])+(1.0/rec[0]))
    fm[1] = 2.0/((1.0/pre[1])+(1.0/rec[1]))
    fm[2] = 2.0/((1.0/pre[2])+(1.0/rec[2]))

    print(pre)
    print(rec)
    print(fm)

''' Experiment Ends here '''

subject= sys.argv[1]
intent= sys.argv[2]
doExperiment(subject)
