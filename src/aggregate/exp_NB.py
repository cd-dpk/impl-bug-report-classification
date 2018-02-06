import numpy as np
import sys,csv
from collections import Counter
from src.aggregate.chou_data import ChouDataHandler

ambari = 'ambari'
camel = 'camel'
derby = 'derby'
wicket = 'wicket'
all = 'all'
Surprising = 'Surprising'
Security = 'Security'
Performance = 'Performance'


subject = ''
intent = ''

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
    chou_data = ChouDataHandler()
    chou_data.load_data(file)

    print(chou_data.textual_data)
    print(Counter(chou_data.target_data))

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.naive_bayes import GaussianNB
    X_text_features = chou_data.textual_data
    y_target = chou_data.target_data
    X_reporter = []
    for x in chou_data.reporter_to_numeric_data():
        X_reporter.append(np.array([x],dtype=int))
    X_reporter = np.array(X_reporter)
    print(X_reporter)
    X_component = chou_data.component_to_numeric_data()

    #X = SelectKBest(chi2(X,y),k=200)

    X_text_features_folds = np.array_split(X_text_features, 10)
    y_target_folds = np.array_split(y_target, 10)
    X_reporter_folds = np.array_split(X_reporter, 10)
    X_component_folds = np.array_split(X_component, 10)
    t_p = 0.0
    f_p = 0.0
    t_n = 0.0
    f_n = 0.0

    h = MultinomialNB()
    for k in range(10):
        # We use 'list' to copy, in order to 'pop' later on
        X_train_text = list(X_text_features_folds)
        X_test_text = X_train_text.pop(k)
        X_train_text = np.concatenate(X_train_text)

        X_train_reporter = list(X_reporter_folds)
        X_test_reporter = X_train_reporter.pop(k)
        X_train_reporter = np.concatenate(X_train_reporter)

        print(X_train_reporter)

        X_train_component = list(X_component_folds)
        X_test_component = X_train_component.pop(k)
        X_train_component = np.concatenate(X_train_component)

        y_train = list(y_target_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)

        X_s, y_s = under_sampling(X_train_text, y_train)
        #X_s, y_s = over_sampling(X_train, y_train)
        #X_s, y_s = smote(X_train, y_train)
        h.fit(X_s,y_s)
        y1_predict = h.predict_proba(X_test_text)

        X_s, y_s = under_sampling(X_train_component, y_train)
        h.fit(X_s,y_s)
        y2_predict = h.predict_proba(X_test_component)

        X_s, y_s = under_sampling(X_train_reporter, y_train)
        h.fit(X_s, y_s)
        y3_predict = h.predict_proba(X_test_reporter)

        y_predict = np.empty(len(y1_predict),int)

        for x in range(len(y_predict)):
            if y1_predict[x][1]+y2_predict[x][1]+y3_predict[x][1] >= y1_predict[x][0]+y2_predict[x][0]+y3_predict[x][0]:
                y_predict[x] = 1
            else:
                y_predict[x] = 0

        print(y_predict)
        print(confusion_matrix(y_test, y_predict))
        temp_tp, temp_tn, temp_fp, temp_fn = calc_tuple(confusion_matrix(y_test, y_predict))
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
