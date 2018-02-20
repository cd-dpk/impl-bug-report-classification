import numpy as np
from collections import Counter
import sys, csv


class BugLocator:

    def __init__(self, file, intent):
        self.file = file
        self.intent = intent
        self.textual_features = []
        self.class_ids = []
        self.class_names = []
        self.class_target = []


    def load_src_data(self):
        with open(self.file+'_vec.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            counter = 0
            for f in reader.fieldnames:
                if counter < 2 or counter == len(reader.fieldnames)-2:
                    counter += 1
                    continue
                else:
                    self.textual_features.append(f)
                    counter += 1

            # print(text_features)
            counter = 0
            for row in reader:
                id = row['file_no']
                name = row['file_name']
                self.class_ids.append(id)
                self.class_names.append(name)
                self.class_target(0)

