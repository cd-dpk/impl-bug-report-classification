import numpy as np
from collections import Counter
import sys, csv


class ChouDataHandler:

    def __init__(self, data_path, file, intent):
        self.data_path = data_path
        self.file = file
        self.intent = intent
        self.reporter_data = []
        self.team_data = []
        self.component_data = []
        self.grep_data = []
        self.lexicon_data = []
        self.textual_data = []
        self.description_data = []
        self.txt_target_data = []
        self.str_target_data = []
        self.text_features = []
        self.str_features = []

    def load_txt_data(self, word2vec: bool= False, dim: int = 200, src: bool= False, des:bool=False):
        self.text_features = []
        self.target_column = ''
        file_name = self.file + '_' + str(word2vec)+ '_' + str(dim) + '_' + str(src) + '_' + str(des)
        print('File Name'+file_name)

        with open(self.data_path + file_name+'_vec.csv', newline='', encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile)
            counter = 0
            for f in reader.fieldnames:
                if counter <= 0 or counter >= len(reader.fieldnames)-2:
                    counter += 1
                    continue
                else:
                    self.text_features.append(f)
                    counter += 1

            if self.intent == 'Security':
                self.target_column = reader.fieldnames[len(reader.fieldnames)-2]
            elif self.intent == 'Performance':
                self.target_column = reader.fieldnames[len(reader.fieldnames) - 1]
            # print(text_features)
            counter = 0
            for row in reader:
                text_data_arr_row = []
                for x in self.text_features:
                    if row[x] not in (None, ''):
                        text_data_arr_row.append(float(row[x]))
                # print(text_data_arr_row, row[self.target_column])
                target = int(row[self.target_column])
                self.textual_data.append(text_data_arr_row)
                self.txt_target_data.append(target)
                counter += 1

        self.textual_data = np.array(self.textual_data)
        self.txt_target_data = np.array(self.txt_target_data)
        return

    def load_str_data(self):
        self.target_column = ''
        file_name = self.file+'_str'

        with open(self.data_path + file_name+'_vec.csv', newline='', encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile)
            grep_key = ''
            if self.intent == 'Security':
                self.target_column = reader.fieldnames[len(reader.fieldnames) - 2]
                grep_key = 'grep_sec'
            elif self.intent == 'Performance':
                self.target_column = reader.fieldnames[len(reader.fieldnames) - 1]
                grep_key = 'grep_perf'
            # print(text_features)
            counter = 0
            for row in reader:
                reporter = row['reporter_col'] in (None, '') and '' or row['reporter_col']
                team = row['team_col'] in (None, '') and '' or row['team_col']
                component = row['component_col'] in (None, '') and 'null' or row['component_col']
                pos = float(row[self.intent+'_pos_col'] in (None, '') and '0' or row[self.intent+'_pos_col'])
                neu = float(row[self.intent+'_neu_col'] in (None, '') and '0' or row[self.intent+'_neu_col'])
                neg = float(row[self.intent+'_neg_col'] in (None, '') and '0' or row[self.intent+'_neg_col'])
                grep = row[grep_key] in (None, '') and '0' or row[grep_key]
                st = ((row['ST_col'] in (None, '') and '0' or row['ST_col']))
                patch = ((row['Patch_col'] in (None, '') and '0' or row['Patch_col']))
                ce = ((row['CE_col'] in (None, '') and '0' or row['CE_col']))
                tc = ((row['TC_col'] in (None, '') and '0' or row['TC_col']))
                en = ((row['EN_col'] in (None, '') and '0' or row['EN_col']))
                # print(counter,reporter,component,keyword,st,patch,ce,tc,en)
                self.reporter_data.append(reporter)
                self.team_data.append(team)
                self.component_data.append(component)
                self.grep_data.append(grep)
                self.description_data.append(np.array([int(st), int(patch), int(ce), int(tc), int(en)]))
                self.lexicon_data.append([pos, neu, neg])
                target = int(row[self.target_column])
                self.str_target_data.append(target)
                counter += 1

        self.reporter_data = np.array(self.reporter_data)
        self.team_data = np.array(self.team_data)
        self.component_data = np.array(self.component_data)
        self.grep_data = np.array(self.grep_data, dtype=int)
        self.description_data = np.array(self.description_data)
        self.str_target_data = np.array(self.str_target_data)
        self.lexicon_data = np.array(self.lexicon_data)

        component_data = np.array(self.component_to_numeric_data())
        self.str_features.append("auth")
        self.str_features.append("team")
        for y in range(len(component_data[0])):
            self.str_features.append("comp"+str(y))

        self.str_features.append("grep")
        for y in range(len(self.lexicon_data[0])):
            self.str_features.append("lex" + str(y))
        for y in range(len(self.description_data[0])):
            self.str_features.append("des" + str(y))
        self.str_features = np.array(self.str_features,dtype=object)

        return

    def load_raw_data(self):
        file_name = self.file
        self.raw_features = []
        self.raw_data = []
        self.raw_target = []
        with open('../data/' + file_name+'.csv', newline='', encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                issue_id = str(row['issue_id'] in (None, '') and '' or row['issue_id'])
                summary = (row['summary'] in (None, '') and '' or row['summary'])
                description = (row['description'] in (None, '') and '' or row['description'])
                security_label = (row['Security'] in (None, '') and '0' or row['Security'])
                perf_label = (row['Performance'] in (None, '') and '0' or row['Performance'])
                self.raw_data.append([issue_id, summary, description])
                if self.intent == 'Security':
                    self.raw_target.append(security_label)
                elif self.intent == 'Performance':
                    self.raw_target.append(perf_label)

        self.raw_data = np.array(self.raw_data)
        self.raw_target = np.array(self.raw_target, dtype=int)
        self.raw_features.append('issue_id')
        self.raw_features.append('summary')
        self.raw_features.append('description')
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


    def get_numeric_str_data(self):
        numeric_data = []
        reporter_data = np.array(self.reporter_to_numeric_data())
        component_data = np.array(self.component_to_numeric_data())

        for x in range(len(self.reporter_data)):
            temp_arr =[]
            temp_arr.append(reporter_data[x])
            temp_arr.append(self.team_data[x])
            for y in range(len(component_data[x])):
                temp_arr.append(component_data[x][y])
            temp_arr.append(self.grep_data[x])

            for y in range(len(self.lexicon_data[x])):
                temp_arr.append(self.lexicon_data[x][y])
            for y in range(len(self.description_data[x])):
                temp_arr.append(self.description_data[x][y])

            numeric_data.append(temp_arr)

        return np.array(numeric_data, dtype=float)

