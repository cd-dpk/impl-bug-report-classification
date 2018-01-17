import numpy as np
import csv
import sys

chou_data = {}
feature_names = 'feature_names'
target_names = 'target_names'
target = 'target'
data = 'data'

data_arr = np.array([],dtype=int)
target_arr = np.array([],dtype=int)

def setFeatureNamesAndRows(file):
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        row_count = 0
        for row in reader:
            row_count += 1

        feature_count = len(reader.fieldnames)-1

        ## set features names

        feature_names_arr = np.empty(feature_count,dtype=object)
        counter = 0
        for x in reader.fieldnames:
            if (len(reader.fieldnames)-1==counter):
                break
            feature_names_arr[counter] = x
            counter +=1

        chou_data[feature_names] = feature_names_arr
        print(chou_data[feature_names])


        # #initialize empty np_array(data,target) with shape of row_count and feature_count

        data_arr = np.empty([row_count,feature_count])
        target_arr = np.empty([row_count])



def load_chou_data(file):
    setFeatureNamesAndRows(file)
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        #set target names
        target_names_arr = np.array(['Surprise','NotSurprise'])
        chou_data[target_names]=target_names_arr

        rw_counter = 0
        # set data and target
        for row in reader:
            f_counter = 0
            data_arr_row = np.empty([len(chou_data[feature_names])],dtype=int)
            features = chou_data[feature_names]
            for x in features:
                data_arr_row[f_counter] = row[f_counter]
                f_counter += 1

            data_arr[rw_counter]= data_arr_row
            target_arr[rw_counter] = row['Surprising']
            rw_counter += 1

        chou_data[data] = data_arr
        chou_data[target] = target_arr

    return chou_data

def process():
    file = 'compsac16/chou_data.csv'
    print(load_chou_data(file))
    #setFeatureNamesAndRows(file)
process()

