import numpy as np
from collections import Counter
import sys, csv



class JiraDataHandler:

    textual_data = []
    target_data = []

    def __init__(self, file, intent):
        self.file = file
        self.intent = intent
        self.text_features = []
        self.target_column = 'target'

    def load_data_featured(self, l, l1_ratio):
        l1 = int(l1_ratio * l)
        l2 = l - l1
        pos_csvfile = open('/media/geet/Files/IITDU/MSSE-03/implementation/src/jira/apache_'+self.intent+'_pos_terms.txt',
                           newline='')
        pos_reader = csv.DictReader(pos_csvfile)
        counter = 0
        for row in pos_reader:
            if counter < l1:
                self.text_features.append((row['term']))
            counter += 1
        pos_csvfile.close()

        neg_csvfile = open('/media/geet/Files/IITDU/MSSE-03/implementation/src/jira/apache_'+self.intent+'_neg_terms.txt',
                       newline='')
        neg_reader = csv.DictReader(neg_csvfile)

        counter = 0
        for row in neg_reader:
            if counter < l2:
                self.text_features.append((row['term']))
            counter += 1
        neg_csvfile.close()

        with open(self.file+'_vec.csv', newline='') as csvfile:

            reader = csv.DictReader(csvfile)

            if self.intent == 'Security':
                self.target_column = reader.fieldnames[len(reader.fieldnames)-2]
            elif self.intent == 'Performance':
                self.target_column = reader.fieldnames[len(reader.fieldnames) - 1]

            print(len(self.text_features))
            print(self.target_column)
            for row in reader:
                text_data_arr_row = []
                for x in self.text_features:
                    if row[x] not in (None, ''):
                        text_data_arr_row.append(float(row[x]))
                target = int(row[self.target_column])
                print(text_data_arr_row, target)
                self.textual_data.append(text_data_arr_row)
                self.target_data.append(target)

        self.textual_data = np.array(self.textual_data)
        self.target_data = np.array(self.target_data)

        return

    def load_data(self):

        with open(self.file+'_vec.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            counter = 0
            for f in reader.fieldnames:
                if counter < (len(reader.fieldnames)-2):
                     self.text_features.append(f)
                counter += 1

            if self.intent == 'Security':
                self.target_column = reader.fieldnames[len(reader.fieldnames)-2]
            elif self.intent == 'Performance':
                self.target_column = reader.fieldnames[len(reader.fieldnames) - 1]

            print(len(self.text_features))
            print(self.target_column)
            # checkpoint = 0
            for row in reader:
                # if checkpoint >=500:
                #     break
                text_data_arr_row = []
                for x in self.text_features:
                    if row[x] not in (None, ''):
                        text_data_arr_row.append(float(row[x]))
                target = int(row[self.target_column])
                # print(text_data_arr_row,target)
                self.textual_data.append(text_data_arr_row)
                self.target_data.append(target)
                # checkpoint += 1

        self.textual_data = np.array(self.textual_data)
        self.target_data = np.array(self.target_data)
        return