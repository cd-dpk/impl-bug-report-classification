import numpy as np
import sys,csv
from collections import Counter
ambari = 'ambari'
camel = 'camel'
derby = 'derby'
wicket = 'wicket'
all = 'all'
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
        feature_names_arr = np.array(reader.fieldnames)
        target_column = intent
        if feature_names_arr[len(feature_names_arr) - 1] == target_column:
            feature_names_arr = np.delete(feature_names_arr, len(feature_names_arr) - 1, axis=0)

        row_count = 0
        for row in reader:
            row_count += 1

        chou_data[feature_names] = feature_names_arr
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



def ensemble_confusion_matrix(y_test,y1_predict,y2_predict,y3_predict,y_):
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
def calc_tuple(result_dic:dict):
    return (result_dic['t_p'],result_dic['t_n'],result_dic['f_p'],result_dic['f_n'])


def calc_test(result_dic:dict):
    t_p = result_dic['t_p']
    t_n = result_dic['t_n']
    f_p = result_dic['f_p']
    f_n = result_dic['f_n']
    return (t_p,t_n)

def doExperiment(file):
    print(load_data(file))
    print(Counter(chou_data[target]))

    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import svm
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier

    X = chou_data[data]
    y = chou_data[target]
    X = SelectKBest(chi2, k=200).fit_transform(chou_data[data], y)

    X_folds = np.array_split(X, 10)
    y_folds = np.array_split(y, 10)

    t_p = 0.0
    f_p = 0.0
    t_n = 0.0
    f_n = 0.0

    H = LogisticRegression()
    h1 = RandomForestClassifier(max_depth=2);
    h2 = GaussianNB()
    h3 = KNeighborsClassifier(5, weights='distance')
    h4 = svm.SVC()
    h5 = GradientBoostingClassifier()
    h6 = AdaBoostClassifier()

    counter = 0
    for k in range(10):
        # We use 'list' to copy, in order to 'pop' later on
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)

        X_folds_2 = np.array_split(X_train, 10)
        y_folds_2 = np.array_split(y_train, 10)

        stacking_target = y_train
        stacking_data = np.empty([len(stacking_target),6],dtype=int)
        training_counter = 0

        ## number of second train folds l
        ## generating the test results

        for l in range(10):
            X_train_2 = list(X_folds_2)
            X_test_2 = X_train_2.pop(l)
            X_train_2 = np.concatenate(X_train_2)
            y_train_2 = list(y_folds_2)
            y_test_2 = y_train_2.pop(l)
            y_train_2 = np.concatenate(y_train_2)

            X_s, y_s = under_sampling(X_train_2, y_train_2)
            h1.fit(X_s, y_s)
            y1_predict = h1.predict(X_test_2)
            h2.fit(X_s, y_s)
            y2_predict = h2.predict(X_test_2)
            h3.fit(X_s, y_s)
            y3_predict = h3.predict(X_test_2)
            h4.fit(X_s,y_s)
            y4_predict = h4.predict(X_test_2)
            h5.fit(X_s,y_s)
            y5_predict = h5.predict(X_test_2)
            h6.fit(X_s,y_s)
            y6_predict = h6.predict(X_test_2)

            for row in range(len(y_test_2)):
                stacking_data[training_counter+row][0] = y1_predict[row]
                stacking_data[training_counter+row][1] = y2_predict[row]
                stacking_data[training_counter+row][2] = y3_predict[row]
                stacking_data[training_counter+row][3] = y4_predict[row]
                stacking_data[training_counter+row][4] = y5_predict[row]
                stacking_data[training_counter+row][5] = y6_predict[row]
                stacking_target[training_counter] = y_test_2[row]

            training_counter += len(y_test_2)


        ### stacking traing with first train data ###
        H.fit(stacking_data,stacking_target)

        ## predicting first test data with stacking
        y1_predict = h1.predict(X_test)
        y2_predict = h2.predict(X_test)
        y3_predict = h3.predict(X_test)
        y4_predict = h4.predict(X_test)
        y5_predict = h5.predict(X_test)
        y6_predict = h6.predict(X_test)

        stacking_target_test = y_test
        stacking_data_test = np.empty([len(X_test),6],dtype=int)

        ## setting first test data's prediction
        for x in range(len(y_test)):
            stacking_data_test[x][0] = y1_predict[x]
            stacking_data_test[x][1] = y2_predict[x]
            stacking_data_test[x][2] = y3_predict[x]
            stacking_data_test[x][3] = y4_predict[x]
            stacking_data_test[x][4] = y5_predict[x]
            stacking_data_test[x][5] = y6_predict[x]

        y_predict = H.predict(stacking_data_test)

        print(confusion_matrix(stacking_target_test, y_predict))
        temp_tp, temp_tn, temp_fp, temp_fn = calc_tuple(confusion_matrix(y_test, y_predict))
        t_p += temp_tp
        t_n += temp_tn
        f_p += temp_fp
        f_n += temp_fn

        #threshold += 0.05

    print(Counter(y))
    print(t_p,t_n,f_p,f_n)


''' Experiment Ends here '''

subject= sys.argv[1]
intent= sys.argv[2]
doExperiment(subject)
