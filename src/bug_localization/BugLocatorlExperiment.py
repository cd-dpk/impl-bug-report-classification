from builtins import list

from sklearn.naive_bayes import MultinomialNB

from src.aggregate.experiment import Experiment
from collections import Counter
import numpy as np
import os


class BugLocatorExperiment(Experiment):


    def __init__(self,file):
        self.src_text_features = []
        self.bug_text_features = []
        self.file = file

    def do_experiment_bug_locate(self):
        # set src text feature
        self.src_text_features = []
        # set bug report features
        self.bug_text_features = []
        self.load_data()
        print(self.X_txt.shape)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        print(Counter(self.y))
        for k in range(len(self.X_txt)):
            # retrieve each bug report
            br = self.X_txt[k]
            # retrieve each src file
            src_files = []
            with open("D:/PythonWorld/implementation/implementation/src/bug_localization" + self.file + "_vec.csv", newline='') as src_csvfile:
                reader = csv.DictReader(src_csvfile)
                print('class_id,class_name,class_content')
                for row in reader:
                    class_id = str(row['file_no'] in (None, '') and '' or row['file_no'])
                    clas_content = row['proc'] in (None, '') and ' ' or row['proc']
                    class_name = row['path'] in (None, '') and ' ' or row['path']

            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))

            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return

