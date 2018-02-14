import numpy as np
from collections import Counter
import sys, csv


class ChouDataHandler:

    reporter_data = []
    component_data = []
    keywords_data = []
    textual_data = []
    description_data = []
    target_data = []

    def __init__(self,file,intent):
        self.file = file
        self.intent = intent


    def set_feature_names_rows(self):
        with open(self.file+'_'+self.intent+ '_vec.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            text_feature_names_arr = np.array(reader.fieldnames)
            target_column = text_feature_names_arr[len(text_feature_names_arr) - 1]
            text_feature_names_arr = np.delete(text_feature_names_arr, len(text_feature_names_arr) - 1, axis=0)
            row_count = len(list(reader))
            self.chou_data[self.feature_names] = text_feature_names_arr
            self.data_arr = np.empty([row_count, len(text_feature_names_arr)], dtype=str)
            self.target_arr = np.empty([row_count], dtype=str)
            self.chou_data[self.data] = self.data_arr
            self.chou_data[self.target] = self.target_arr
        return

    def load_data(self):
        with open(self.file+'_'+self.intent+'_vec.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            text_features = []
            target_column=''
            counter = 0
            for f in reader.fieldnames:
                if counter <= 7 or counter == len(reader.fieldnames)-1:
                    counter += 1
                    continue
                else:
                    text_features.append(f)
                    counter += 1

            target_column = reader.fieldnames[len(reader.fieldnames)-1]
            # print(text_features)

            for row in reader:
                reporter = row['reporter'] in (None, '') and '' or row['reporter']
                component = row['component'] in (None, '') and 'null' or row['component']
                keyword = row['keywords'] in (None, '') and '0' or row['keywords']
                st = ((row['des-1'] in (None, '') and '0' or row['des-1']))
                patch = ((row['des-2'] in (None, '') and '0' or row['des-2']))
                ce = ((row['des-3'] in (None, '') and '0' or row['des-3']))
                tc = ((row['des-4'] in (None, '') and '0' or row['des-4']))
                en = ((row['des-5'] in (None, '') and '0' or row['des-5']))
                # print(reporter,component,st,patch,ce,tc,en)
                self.reporter_data.append(reporter)
                self.component_data.append(component)
                self.keywords_data.append(int(keyword))
                self.description_data.append(np.array([st, patch, ce, tc, en]).astype(int))
                # print(self.description_data)
                text_data_arr_row = []
                for x in text_features:
                    if row[x] not in (None, ''):
                        text_data_arr_row.append(float(row[x]))

                target = int(row[target_column])
                self.textual_data.append(text_data_arr_row)
                self.target_data.append(target)

        self.reporter_data = np.array(self.reporter_data)
        self.component_data = np.array(self.component_data)
        self.keywords_data = np.array(self.keywords_data)
        self.textual_data = np.array(self.textual_data)
        self.description_data = np.array(self.description_data)
        self.target_data = np.array(self.target_data)
        return

    def reporter_to_numeric_data(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(self.reporter_data)
        return np.array(le.transform(self.reporter_data)).astype(int)

    def component_to_numeric_data(self):
        import re
        component_list = []
        # print(self.component_data)
        for x in range(len(self.component_data)):
            # print(self.component_data[x])
            comps = re.split("; ", self.component_data[x][0])
            for x_comp in comps:
               if x_comp not in component_list:
                component_list.append(x_comp)

        one_hot_components = []
        for x in range(len(self.component_data)):
            comps = re.split("; ", self.component_data[x][0])
            one_hot_component = []
            for c in component_list:
                if c in comps:
                   one_hot_component.append(1)
                else:
                    one_hot_component.append(0)
            one_hot_components.append(np.array(one_hot_component))

        return np.array(one_hot_components).astype(int)


    def get_numeric_data(self):
        numeric_data = []
        reporter_data = np.array(self.reporter_to_numeric_data())
        component_data = np.array(self.component_to_numeric_data())

        for x in range(len(self.reporter_data)):
            # print((reporter_data[0], component_data[0], self.keywords_data[0], self.textual_data[0]))
            row = np.concatenate((reporter_data[x], component_data[x],self.keywords_data[x],self.textual_data[x]),axis=0)
            numeric_data.append(row)

        return np.array(numeric_data).astype(int)


    def get_numeric_str_data(self):
        numeric_data = []
        reporter_data = np.array(self.reporter_to_numeric_data())
        component_data = np.array(self.component_to_numeric_data())
        for x in range(len(self.reporter_data)):
            temp_arr = []
            temp_arr.append(reporter_data[x])
            for y in range(len(component_data[x])):
                temp_arr.append(component_data[x][y])
            temp_arr.append(self.keywords_data[x])
            for y in range(len(self.description_data[x])):
                temp_arr.append(self.description_data[x][y])
            numeric_data.append(temp_arr)
        return np.array(numeric_data)