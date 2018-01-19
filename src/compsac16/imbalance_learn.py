import numpy as np
import csv
import sys
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

chou_data = {}
feature_names = 'feature_names'
target_names = 'target_names'
target = 'target'
data = 'data'

data_arr = np.array([], dtype=str)
target_arr = np.array([], dtype=str)


def setFeatureNamesAndRows(file):
    with open(file, newline='') as csvfile:
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


def load_chou_data(file):
    setFeatureNamesAndRows(file)
    # print(chou_data[feature_names])
    with open(file, newline='') as csvfile:
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


def process():
    file = 'ambari_data.csv'
    ambari = load_chou_data(file)
    y = ambari[target]
    x = ambari[data]
    print(Counter(y))

    # Apply the random under-sampling
    print('----------------Under Sampling-----------------')
    rus = RandomUnderSampler()
    X_resampled, y_resampled = rus.fit_sample(x, y)
    print(Counter(y_resampled))
    gnb = MultinomialNB()
    scores = cross_val_score(gnb, X_resampled, y_resampled, cv=10)
    print(scores)

    # Appy the random over-sampling
    print('----------------Over Sampling-----------------')
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_sample(x, y)
    print(Counter(y_resampled))
    gnb = MultinomialNB()
    scores = cross_val_score(gnb, X_resampled, y_resampled, cv=10)
    print(scores)

    # Appy the SMOTE
    print('---------------- SMOTE -----------------')
    sm = SMOTE()
    X_resampled, y_resampled = sm.fit_sample(x, y)
    print(Counter(y_resampled))
    gnb = MultinomialNB()
    scores = cross_val_score(gnb, X_resampled, y_resampled, cv=10)
    print(scores)

process()

