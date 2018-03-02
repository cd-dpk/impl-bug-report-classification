import numpy as np
from collections import Counter
import sys, csv


class ChouDataHandler:
    def __init__(self, file, intent):
        self.file = file
        self.intent = intent
        self.reporter_data = []
        self.component_data = []
        self.lexicon_data = []
        self.textual_data = []
        self.description_data = []
        self.target_data = []


    def load_data(self, word2vec: bool):
        file_name = self.file
        if word2vec == True:
            file_name += 'wv'

        with open(file_name+'_vec.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            text_features = []
            target_column= ''
            counter = 0
            for f in reader.fieldnames:
                if counter <= 12 or counter == len(reader.fieldnames)-2:
                    counter += 1
                    continue
                else:
                    text_features.append(f)
                    counter += 1

            if self.intent == 'Security':
                self.target_column = reader.fieldnames[len(reader.fieldnames)-2]
            elif self.intent == 'Performance':
                self.target_column = reader.fieldnames[len(reader.fieldnames) - 1]
            # print(text_features)
            counter = 0
            for row in reader:
                reporter = row['reporter_col'] in (None, '') and '' or row['reporter_col']
                component = row['component_col'] in (None, '') and 'null' or row['component_col']
                pos = float(row[self.intent+'_pos_col'] in (None, '') and '0' or row[self.intent+'_pos_col'])
                neu = float(row[self.intent+'_neu_col'] in (None, '') and '0' or row[self.intent+'_neu_col'])
                neg = float(row[self.intent+'_neg_col'] in (None, '') and '0' or row[self.intent+'_neg_col'])
                # keyword_sec = row['keyword_sec_col'] in (None, '') and '0' or row['keyword_sec_col']
                # keyword_perf = row['keyword_perf_col'] in (None, '') and '0' or row['keyword_perf_col']
                # # print(keyword_sec, keyword_perf)
                st = ((row['ST_col'] in (None, '') and '0' or row['ST_col']))
                patch = ((row['Patch_col'] in (None, '') and '0' or row['Patch_col']))
                ce = ((row['CE_col'] in (None, '') and '0' or row['CE_col']))
                tc = ((row['TC_col'] in (None, '') and '0' or row['TC_col']))
                en = ((row['EN_col'] in (None, '') and '0' or row['EN_col']))
                # print(counter,reporter,component,keyword,st,patch,ce,tc,en)
                self.reporter_data.append(reporter)
                self.component_data.append(component)
                # self.keyword_sec_data.append(int(keyword_sec))
                # self.keyword_perf_data.append(int(keyword_perf))
                self.description_data.append(np.array([int(st), int(patch), int(ce), int(tc), int(en)]))
                self.lexicon_data.append([pos, neu, neg])
                text_data_arr_row = []
                for x in text_features:
                    if row[x] not in (None, ''):
                        text_data_arr_row.append(float(row[x]))
                # print(text_data_arr_row, row[self.target_column])
                target = int(row[self.target_column])
                self.textual_data.append(text_data_arr_row)
                self.target_data.append(target)
                counter += 1

        self.reporter_data = np.array(self.reporter_data)
        self.component_data = np.array(self.component_data)
        # self.keyword_sec_data = np.array(self.keyword_sec_data)
        # self.keyword_perf_data = np.array(self.keyword_perf_data)
        self.textual_data = np.array(self.textual_data)
        self.description_data = np.array(self.description_data)
        self.target_data = np.array(self.target_data)
        self.lexicon_data = np.array(self.lexicon_data)
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

            for y in range(len(self.lexicon_data[x])):
                temp_arr.append(self.lexicon_data[x][y])

            for y in range(len(self.description_data[x])):
                temp_arr.append(self.description_data[x][y])
            numeric_data.append(temp_arr)
        return np.array(numeric_data)