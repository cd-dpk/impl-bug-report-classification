from builtins import list
import re
from sklearn.naive_bayes import MultinomialNB

from src.aggregate.experiment import Experiment
from collections import Counter
import numpy as np
import os, csv

from src.aggregate.feature_selection import FeatureSelector

class NormalExperiment(Experiment):

    def do_experiment_txt_str(self, hypo):
        # classify bug report through text using feature selection
        # classify the structures
        self.load_data()
        print(self.X_txt)
        print(self.X_str)

        X_txt_folds = np.array_split(self.X_txt, 10)
        X_str_folds = np.array_split(self.X_str, 10)
        y_folds = np.array_split(self.y, 10)

        self.X_txt = FeatureSelector().fit_transform_odd_ratio(self.X_txt, self.y, 400, 0.5)

        from sklearn.feature_selection import SelectFdr, chi2
        ch2 = SelectFdr(score_func=chi2, alpha=0.01)
        self.X_str = ch2.fit_transform(self.X_str, self.y)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        print(Counter(self.y))
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_txt_train = list(X_txt_folds)
            X_txt_test = X_txt_train.pop(k)
            X_txt_train = np.concatenate(X_txt_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)
            X_str_train = list(X_str_folds)
            X_str_test = X_str_train.pop(k)
            X_str_train = np.concatenate(X_str_train)

            y_predict = np.zeros(len(y_test), dtype=int)

            # deal with prediction
            '''
            hypo.fit(X_txt_train, y_train)
            y_txt_predict = hypo.predict(X_txt_test)

            X_str_train, y_train = self.smote(X_str_train,y_train)
            hypo.fit(X_str_train, y_train)
            y_str_predict = hypo.predict(X_str_test)

            for y in range(len(y_test)):
                if y_str_predict[y] == 0 and y_txt_predict[y] == 0:
                    y_predict[y] = 0
                else:
                    y_predict[y] = 1
                print(y_txt_predict[y],y_str_predict[y],y_predict[y])

            '''

            # deal with probability
            # '''
            hypo.fit(X_txt_train, y_train)
            y_txt_predict_prob = hypo.predict_proba(X_txt_test)

            X_str_train, y_train = self.under_sampling(X_str_train, y_train)
            hypo.fit(X_str_train, y_train)
            y_str_predict_prob = hypo.predict_proba(X_str_test)

            for y in range(len(y_test)):
                zero_proba = y_txt_predict_prob[y][0] + y_str_predict_prob[y][0]
                one_proba = y_txt_predict_prob[y][1] + y_str_predict_prob[y][1]
                if one_proba >= zero_proba:
                    y_predict[y] = 1
                else:
                    y_predict[y] = 0
            # '''

            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return


    def do_experiment_txt_feature_selection(self, l, l1_ratio,hypo):
        self.load_data()
        print(self.X_txt.shape)
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
        from src.aggregate.feature_selection import FeatureSelector
        self.X_txt = FeatureSelector().fit_transform_odd_ratio(self.X_txt, self.y, l, l1_ratio)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        print(Counter(self.y))

        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)
            hypo.fit(X_train, y_train)
            y_predict = hypo.predict(X_test)
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))

            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn


        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return

    def do_experiment_txt_sampling_ensemble_stacking(self, sampling_index:int, Hypo, hypos:list):
        self.load_data()
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        len_hypos = len(hypos)

        for l in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(l)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(l)
            y_train = np.concatenate(y_train)
            fold = 10
            X_folds_2 = np.array_split(X_train, fold)
            y_folds_2 = np.array_split(y_train, fold)

            ## number of second train folds l
            '''
                Generating the test results
                and stores in the y_predicts_prob

            '''

            y_predicts_proba = np.zeros([len(X_train), 2 * len_hypos], dtype=float)
            print(y_predicts_proba.shape)
            row = 0
            for k in range(fold):
                X_train_2 = list(X_folds_2)
                X_test_2 = X_train_2.pop(k)
                X_train_2 = np.concatenate(X_train_2)
                y_train_2 = list(y_folds_2)
                y_test_2 = y_train_2.pop(k)
                y_train_2 = np.concatenate(y_train_2)
                X_train_2, y_train_2 = self.smote(X_train_2, y_train_2)

                column = 0
                for hypo in hypos:
                    hypo.fit(X_train_2, y_train_2)
                    y_predict_proba = hypo.predict_proba(X_test_2)

                    for x in range(len(y_predict_proba)):
                        print(row + x, 2*column+0)
                        print(row + x, 2*column+1)
                        y_predicts_proba[row + x][2*column+0] = y_predict_proba[x][0]
                        y_predicts_proba[row + x][2*column+1] = y_predict_proba[x][1]

                    column += 1

                row += len(X_test_2)



            '''
                Stacking training with first train data probabilities
            '''
            print(y_predicts_proba)
            print(y_predicts_proba.shape)
            Hypo.fit(y_predicts_proba, y_train)

            '''
                Predicting first test data with stacking
            '''

            y_predicts_proba = np.empty([len(X_test), 2 * len_hypos], dtype=int)

            column = 0
            for hypo in hypos:
                y_predict_proba = hypo.predict_proba(X_test)
                for x in range(len(y_predict_proba)):
                    y_predicts_proba[x][2*column+0] = y_predict_proba[x][0]
                    y_predicts_proba[x][2*column+1] = y_predict_proba[x][1]
                column += 1

            y_predict = Hypo.predict(y_predicts_proba)

            # print(self.confusion_matrix(y_test, y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return

    def do_experiment_txt_sampling_ensemble_voting(self,sampling_index:int,hypos:list):
        self.load_data()
        print(self.X_txt.shape)
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        len_of_hypos = len(hypos)
        print(Counter(self.y))
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)
            X_s, y_s = self.smote(X_train, y_train)
            if sampling_index == 0:
                X_s, y_s = self.under_sampling(X_train, y_train)
            elif sampling_index == 1:
                X_s, y_s = self.over_sampling(X_train, y_train)

            y_predicts = np.empty([len(y_test), len_of_hypos], dtype=int)
            column = 0
            for hypo in hypos:
                hypo.fit(X_s, y_s)
                print(hypo.predict_proba(X_test))
                y_predict = hypo.predict(X_test)
                for x in range(len(y_predict)):
                    y_predicts[x][column] = y_predict[x]

                column += 1

            y_predict = np.empty(len(y_test), dtype=int)
            for r in range(len(y_predict)):
                zeros = 0
                ones = 0

                for h in range(len_of_hypos):
                    if y_predicts[r][h] == 0:
                        zeros += 1
                    elif y_predicts[r][h] == 1:
                        ones += 1

                if ones >= zeros:
                    y_predict[r] = 1
                else:
                    y_predict[r] = 0

            #   if y_predict[r] == 1 or y_test[r] == 1:
            #   print(y_predicts[r], ones, zeros, y_predict[r], y_test[r])

            # print(self.confusion_matrix(y_test,y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return

    def do_experiment_txt_sampling_ensemble_probability_voting(self,sampling_index:int,hypos:list):
        self.load_data()
        print(self.X_txt.shape)
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        len_of_hypos = len(hypos)
        print(Counter(self.y))
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)
            X_s, y_s = self.smote(X_train, y_train)
            if sampling_index == 0:
                X_s, y_s = self.under_sampling(X_train, y_train)
            elif sampling_index == 1:
                X_s, y_s = self.over_sampling(X_train, y_train)

            y_predicts = np.empty([len(y_test), 2 * len_of_hypos], dtype=int)
            column = 0
            for hypo in hypos:
                hypo.fit(X_s, y_s)
                print(hypo.predict_proba(X_test))
                y_predict_proba = hypo.predict_proba(X_test)
                for x in range(len(y_test)):
                    y_predicts[x][2*column + 0] = y_predict_proba[x][0]
                    y_predicts[x][2*column + 1] = y_predict_proba[x][1]
                column += 1

            y_predict = np.empty(len(y_test), dtype=int)
            for r in range(len(y_predict)):
                zeros = 0.0
                ones = 0.0
                for h in range(len_of_hypos):
                    zeros += y_predicts[r][2*h+0]
                    ones += y_predicts[r][2*h+1]

                if ones >= zeros:
                    y_predict[r] = 1
                else:
                    y_predict[r] = 0

            #   if y_predict[r] == 1 or y_test[r] == 1:
            #   print(y_predicts[r], ones, zeros, y_predict[r], y_test[r])

            # print(self.confusion_matrix(y_test,y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return


    def do_experiment_txt_sampling_classifier(self, sampling_index=0, hypo=MultinomialNB()):
        self.load_data()
        print(self.X_txt.shape)
        print(self.X_txt)
        print(self.y)
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        print(Counter(self.y))

        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            if sampling_index == 0 :
                X_s, y_s = self.under_sampling(X_train, y_train)
                hypo.fit(X_s, y_s)
            elif sampling_index == 1:
                X_s, y_s = self.over_sampling(X_train, y_train)
                hypo.fit(X_s, y_s)
            else:
                X_s, y_s = self.smote(X_train, y_train)
                hypo.fit(X_s, y_s)

            y_predict = hypo.predict(X_test)
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))

            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn


        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return

    def do_experiment_tex_src(self, sampling_index, hypo):
        self.load_data()
        print(self.X_txt.shape)
        print(self.X_txt)
        print(self.y)
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        print(Counter(self.y))

        # load all bug reports
        bug_reports = []
        with open(self.file + "_proc.csv", newline='') as bug_csvfile:
            bug_reader = csv.DictReader(bug_csvfile)
            for bug_row in bug_reader:
                summary = str(bug_row['summary'] in (None, '') and '' or bug_row['summary'])
                description = str(bug_row['description'] in (None, '') and '' or bug_row['description'])
                files = str(bug_row['files'] in (None, '') and '' or bug_row['files'])
                target_Security = int(bug_row['target_Security'] in (None, '') and '' or bug_row['target_Security'])
                target_Performance = int(bug_row['target_Performance'] in (None, '') and '' or bug_row['target_Performance'])
                bug_report = {'summary': summary, 'description':description, 'files': files, 'target_Security':target_Security, 'target_Performance':target_Performance }
                bug_reports.append(bug_report)

        # for bug_report in bug_reports:
        #     print(bug_report)

        X_bug_reports_folds = np.array_split(bug_reports, 10)

        src_files = []
        with open('/media/geet/Files/IITDU/MSSE-03/implementation/src/bug_localization/'+self.file + "_proc.csv", newline='') as src_csvfile:
            src_reader = csv.DictReader(src_csvfile)
            for src_row in src_reader:
                class_id = int(src_row['class_id'] in (None, '') and '' or src_row['class_id'])
                class_name = str(src_row['class_name'] in (None, '') and '' or src_row['class_name'])
                class_label = 0
                # print(class_obj)
                if re.search('^/media/geet/Files/IITDU/MSSE-03/SRC/', class_name) and re.search('.java$', class_name):
                    class_name = re.sub('/media/geet/Files/IITDU/MSSE-03/SRC/', "", class_name)

                class_obj = {'class_id':class_id, 'class_name': class_name, 'class_label':class_label}
                src_files.append(class_obj)


        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            X_bug_reports_train = list(X_bug_reports_folds)
            X_bug_reports_test = X_bug_reports_train.pop(k)
            X_bug_reports_train = np.concatenate(X_bug_reports_train)

            for i in range(len(src_files)):
                src_files[i][2] = 0

            print(Counter(y_train))
            for bug_report in X_bug_reports_train:
                file_column = bug_report['files']
                files = re.split(";", file_column)
                if bug_report['target_Security'] == 1:
                    for file in files:
                        file = "camel/"+file
                        if re.search(".java$", file):
                            for i in range(len(src_files)):
                                if file == src_files[i]['class_name']:
                                    src_files[i]['class_label'] = 1

            if sampling_index == 0:
                X_s, y_s = self.under_sampling(X_train, y_train)
                hypo.fit(X_s, y_s)
            elif sampling_index == 1:
                X_s, y_s = self.over_sampling(X_train, y_train)
                hypo.fit(X_s, y_s)
            else:
                X_s, y_s = self.smote(X_train, y_train)
                hypo.fit(X_s, y_s)

            y_predict_classifier = hypo.predict(X_test)
            y_predict_localization = np.zeros(len(X_test),dtype=int)

            from src.bug_localization.BugLocatorlExperiment import BugLocatorExperiment
            bug_locator = BugLocatorExperiment(self.file, self.intent)

            for i in range(len(X_bug_reports_test)):
                ranked_src_files = bug_locator.do_experiment_bug_locate(X_bug_reports_test[i])
                if len(ranked_src_files) == 0: break
                for src_file in src_files:
                    if ranked_src_files[0]['class_name'] == src_file['class_name']:
                        if src_file['class_label'] == 1:
                            y_predict_localization[i] = 1
                        else:
                            y_predict_localization[i] = 0

            y_predict = np.zeros(len(X_test), dtype=int)

            for i in range(len(y_predict)):
                if y_predict_localization[i] == 0 and y_predict_classifier[i] == 0:
                    y_predict[i] = 0
                else:
                    y_predict[i] = 1

            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))

            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return

    def do_experiment_tex_fs_str_sampling_src(self, sampling_index, hypo):
        self.load_data()
        print(self.X_txt.shape)
        print(self.X_str.shape)
        print(self.X_txt)
        print(self.y)
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        print(Counter(self.y))

        # load all bug reports
        bug_reports = []
        with open(self.file + "_proc.csv", newline='') as bug_csvfile:
            bug_reader = csv.DictReader(bug_csvfile)
            for bug_row in bug_reader:
                summary = str(bug_row['summary'] in (None, '') and '' or bug_row['summary'])
                description = str(bug_row['description'] in (None, '') and '' or bug_row['description'])
                files = str(bug_row['files'] in (None, '') and '' or bug_row['files'])
                target_Security = int(bug_row['target_Security'] in (None, '') and '' or bug_row['target_Security'])
                target_Performance = int(bug_row['target_Performance'] in (None, '') and '' or bug_row['target_Performance'])
                bug_report = {'summary': summary, 'description':description, 'files': files, 'target_Security':target_Security, 'target_Performance':target_Performance }
                bug_reports.append(bug_report)

        # for bug_report in bug_reports:
        #     print(bug_report)

        X_bug_reports_folds = np.array_split(bug_reports, 10)

        src_files = []
        with open('/media/geet/Files/IITDU/MSSE-03/implementation/src/bug_localization/'+self.file + "_proc.csv", newline='') as src_csvfile:
            src_reader = csv.DictReader(src_csvfile)
            for src_row in src_reader:
                class_id = int(src_row['class_id'] in (None, '') and '' or src_row['class_id'])
                class_name = str(src_row['class_name'] in (None, '') and '' or src_row['class_name'])
                class_label = 0
                # print(class_obj)
                if re.search('^/media/geet/Files/IITDU/MSSE-03/SRC/', class_name) and re.search('.java$', class_name):
                    class_name = re.sub('/media/geet/Files/IITDU/MSSE-03/SRC/', "", class_name)

                class_obj = {'class_id':class_id, 'class_name': class_name, 'class_label':class_label}
                src_files.append(class_obj)


        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            X_bug_reports_train = list(X_bug_reports_folds)
            X_bug_reports_test = X_bug_reports_train.pop(k)
            X_bug_reports_train = np.concatenate(X_bug_reports_train)

            for i in range(len(src_files)):
                src_files[i][2] = 0

            print(Counter(y_train))
            for bug_report in X_bug_reports_train:
                file_column = bug_report['files']
                files = re.split(";", file_column)
                if bug_report['target_Security'] == 1:
                    for file in files:
                        file = "camel/"+file
                        if re.search(".java$", file):
                            for i in range(len(src_files)):
                                if file == src_files[i]['class_name']:
                                    src_files[i]['class_label'] = 1

            if sampling_index == 0:
                X_s, y_s = self.under_sampling(X_train, y_train)
                hypo.fit(X_s, y_s)
            elif sampling_index == 1:
                X_s, y_s = self.over_sampling(X_train, y_train)
                hypo.fit(X_s, y_s)
            else:
                X_s, y_s = self.smote(X_train, y_train)
                hypo.fit(X_s, y_s)

            y_predict_classifier = hypo.predict(X_test)
            y_predict_localization = np.zeros(len(X_test),dtype=int)

            from src.bug_localization.BugLocatorlExperiment import BugLocatorExperiment
            bug_locator = BugLocatorExperiment(self.file, self.intent)

            for i in range(len(X_bug_reports_test)):
                ranked_src_files = bug_locator.do_experiment_bug_locate(X_bug_reports_test[i])
                if len(ranked_src_files) == 0: break
                for src_file in src_files:
                    if ranked_src_files[0]['class_name'] == src_file['class_name']:
                        if src_file['class_label'] == 1:
                            y_predict_localization[i] = 1
                        else:
                            y_predict_localization[i] = 0

            y_predict = np.zeros(len(X_test), dtype=int)

            for i in range(len(y_predict)):
                if y_predict_localization[i] == 0 and y_predict_classifier[i] == 0:
                    y_predict[i] = 0
                else:
                    y_predict[i] = 1

            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))

            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return

    def do_experiment_src(self):
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0

        # load all bug reports
        bug_reports = []
        with open(self.file + "_proc.csv", newline='') as bug_csvfile:
            bug_reader = csv.DictReader(bug_csvfile)
            for bug_row in bug_reader:
                summary = str(bug_row['summary'] in (None, '') and '' or bug_row['summary'])
                description = str(bug_row['description'] in (None, '') and '' or bug_row['description'])
                files = str(bug_row['files'] in (None, '') and '' or bug_row['files'])
                target_Security = int(bug_row['target_Security'] in (None, '') and '' or bug_row['target_Security'])
                target_Performance = int(bug_row['target_Performance'] in (None, '') and '' or bug_row['target_Performance'])
                bug_report = {'summary': summary, 'description': description, 'files': files,
                              'target_Security': target_Security, 'target_Performance': target_Performance}
                bug_reports.append(bug_report)

        # for bug_report in bug_reports:
        #     print(bug_report)

        X_bug_reports_folds = np.array_split(bug_reports, 10)

        src_files = []
        with open('../bug_localization/' + self.file + "_proc.csv",
                  newline='') as src_csvfile:
            src_reader = csv.DictReader(src_csvfile)
            for src_row in src_reader:
                class_id = int(src_row['class_id'] in (None, '') and '' or src_row['class_id'])
                class_name = str(src_row['class_name'] in (None, '') and '' or src_row['class_name'])
                class_label = 0
                # print(class_obj)
                if re.search('^/media/geet/Files/IITDU/MSSE-03/SRC/', class_name) and re.search('.java$', class_name):
                    class_name = re.sub('/media/geet/Files/IITDU/MSSE-03/SRC/', "", class_name)

                class_obj = {'class_id': class_id, 'class_name': class_name, 'class_label': class_label}
                src_files.append(class_obj)

        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_bug_reports_train = list(X_bug_reports_folds)
            X_bug_reports_test = X_bug_reports_train.pop(k)
            X_bug_reports_train = np.concatenate(X_bug_reports_train)

            for i in range(len(src_files)):
                src_files[i][2] = 0

            for bug_report in X_bug_reports_train:
                file_column = bug_report['files']
                files = re.split(";", file_column)
                if bug_report['target_Security'] == 1:
                    for file in files:
                        file = "camel/" + file
                        if re.search(".java$", file):
                            for i in range(len(src_files)):
                                if file == src_files[i]['class_name']:
                                    src_files[i]['class_label'] = 1


            y_predict_localization = np.zeros(len(X_bug_reports_test), dtype=int)

            from src.bug_localization.BugLocatorlExperiment import BugLocatorExperiment
            bug_locator = BugLocatorExperiment(self.file, self.intent)

            for i in range(len(X_bug_reports_test)):
                ranked_src_files = bug_locator.do_experiment_bug_locate(X_bug_reports_test[i])
                if len(ranked_src_files) == 0:
                    break
                for src_file in src_files:
                    if ranked_src_files[0]['class_name'] == src_file['class_name']:
                        if src_file['class_label'] == 1:
                            y_predict_localization[i] = 1
                        else:
                            y_predict_localization[i] = 0
                        break

            y_predict = np.zeros(len(X_bug_reports_test), dtype=int)

            for i in range(len(y_predict)):
                if y_predict_localization[i] == 0:
                    y_predict[i] = 0
                else:
                    y_predict[i] = 1

            y_test = np.zeros(len(X_bug_reports_test), dtype=int)
            for i in range(len(X_bug_reports_test)):
                y_test[i] = X_bug_reports_test[i]['target_'+self.intent]


            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return
