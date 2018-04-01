from builtins import list
from sklearn.naive_bayes import MultinomialNB
from src.aggregate.experiment import Experiment
from collections import Counter
import numpy as np
import os, csv, re

class NormalExperiment(Experiment):


    def do_experiment_grep(self):
        from src.aggregate.grep import GREP
        self.load_data()
        grep = GREP()
        print(len(self.y_raw))
        print(Counter(self.y_raw))
        y_test = self.y_raw
        y_predict = np.zeros(len(self.y_raw), dtype=int)
        print(len(self.X_raw))
        for x in range(len(self.X_raw)):
            if self.intent == 'Security':
                garbage, y_predict[x] = grep.predict_security_label(self.X_raw[x][1], self.X_raw[x][2])
            elif self.intent == 'Performance':
                garbage, y_predict[x] = grep.predict_performance_label(self.X_raw[x][1], self.X_raw[x][2])

        print(self.calc_pre_rec_acc_fpr_tpr(self.confusion_matrix(y_test, y_predict)))

        return

    def do_experiment_text(self):
        print(self.file, self.intent)
        self.load_data()
        print(self.X_txt)
        print(self.txt_features)
        print(len(self.txt_features))
        print(self.X_str)
        print(self.str_features)
        print(len(self.str_features))
        return

    # @imbalance @sampling @text @single_classifier
    def do_experiment_txt_sampling_classifier(self, des: bool=False, sampling_index: int=0, hypo=MultinomialNB()):
        print(self.file, self.intent)
        self.load_data(des=des)
        print(self.X_txt)
        fold = 10
        X_folds = np.array_split(self.X_txt, fold)
        y_folds = np.array_split(self.y_txt, fold)
        pre, rec, acc, fpr, tpr = 0.0, 0.0, 0.0, 0.0, 0.0
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        print(Counter(self.y_txt))
        logfile = open(self.data_path + self.file + '_' + self.intent + '_' + str(des) + '_' + str(sampling_index) + '_log.txt', 'w')
        for k in range(fold):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)
            print(Counter(y_train), Counter(y_test))
            if sampling_index == 0:
                X_s, y_s = self.under_sampling(X_train, y_train)
                hypo.fit(X_s, y_s)
            elif sampling_index == 1:
                X_s, y_s = self.over_sampling(X_train, y_train)
                hypo.fit(X_s, y_s)
            else:
                X_s, y_s = self.smote(X_train, y_train)
                hypo.fit(X_s, y_s)

            print(Counter(y_s))
            y_predict = hypo.predict(X_test)
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            temp_pre, temp_rec, temp_acc, temp_fpr, temp_tpr = self.calc_pre_rec_acc_fpr_tpr(self.confusion_matrix(y_test, y_predict))

            pre += temp_pre
            rec += temp_rec
            acc += temp_acc
            fpr += temp_fpr
            tpr += temp_tpr

            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        pre /= fold
        rec /= fold
        acc /= fold
        fpr /= fold
        tpr /= fold
        print(t_p, t_n, f_p, f_n)
        logfile.write(str(t_p) + "," + str(t_n) + "," + str(f_p) + "," + str(f_n) + "\n")
        print(pre, rec, acc)
        logfile.write(str(pre) + "," + str(rec ) + "," + str(acc) + "\n")
        print(fpr, tpr)
        logfile.write(str(fpr) + "," + str(tpr) + "\n")
        logfile.close()
        return

    # @text @feature selection
    def do_experiment_txt_feature_selection(self, des:bool= False, hypo=MultinomialNB(), alpha=0.5):
        self.load_data(des=des)
        print(self.X_txt)
        print(self.X_txt.shape)
        total_data, total_features = self.X_txt.shape
        X_folds = np.array_split(self.X_txt, 10)
        self.X_txt = []
        y_folds = np.array_split(self.y_txt, 10)
        self.y_txt = []
        feature_num = [0.15, 0.25, 0.40,
                       0.5, 0.65, 0.75,
                       0.85, 1.0]
        t_p = np.zeros(len(feature_num), dtype=int)
        f_p = np.zeros(len(feature_num), dtype=int)
        t_n = np.zeros(len(feature_num), dtype=int)
        f_n = np.zeros(len(feature_num), dtype=int)
        print(t_p)
        print(Counter(self.y_txt))

        logfile = open(self.data_path + self.file + '_' + self.intent + '_' + str(des) + '_' +str(alpha)+'_log.txt', 'w')
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            print("Before FS", X_train.shape, X_test.shape)
            from src.aggregate.feature_selection import FeatureSelector
            feature_selector = FeatureSelector(selection_method=0)
            feature_selector.fit(X_train, y_train)
            column = 0
            for f_num in feature_num:
                print("Features", f_num)
                X_temp_train = feature_selector.transform(X_train, int(f_num * total_features), alpha)
                X_temp_test = feature_selector.transform(X_test, int(f_num * total_features), alpha)
                print("After FS", X_temp_train.shape, X_temp_test.shape)
                hypo.fit(X_temp_train, y_train)
                y_predict = hypo.predict(X_temp_test)
                temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
                t_p[column] += temp_tp
                t_n[column] += temp_tn
                f_p[column] += temp_fp
                f_n[column] += temp_fn
                column += 1

        for x in range(len(feature_num)):
            print(feature_num[x])
            logfile.write(str(feature_num[x]) + ',' + str(int(feature_num[x] * total_features)) + "\n")
            print(t_p[x], t_n[x], f_p[x], f_n[x])
            logfile.write(str(t_p[x]) + "," + str(t_n[x]) + "," + str(f_p[x]) + "," + str(f_n[x]) + "\n")
            acc, pre, rec = self.calc_acc_pre_rec({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]})
            print(self.calc_acc_pre_rec({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]}))
            logfile.write(str(acc) + "," + str(pre) + "," + str(rec) + "\n")
            fpr, tpr = self.calc_fpr_tpr({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]})
            print(self.calc_fpr_tpr({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]}))
            logfile.write(str(fpr) + "," + str(tpr) + "\n")

        logfile.close()
        return

    # @text @sampling @feature_selection
    def do_experiment_txt_sampling_feature_selection(self, des: bool=False, sampling_index: int=0, hypo=MultinomialNB(),alpha=0.5):
        print(self.intent)
        self.load_data(des=des)
        print('Counter', Counter(self.y_txt))
        # print(self.X_txt)
        print('Shape', self.X_txt.shape)
        fold = 10
        # return
        total_data, total_features = self.X_txt.shape
        X_folds = np.array_split(self.X_txt, fold)
        y_folds = np.array_split(self.y_txt, fold)
        print(Counter(self.y_txt))
        self.X_txt = []
        self.y_txt = []
        print("DIM", total_data, total_features)
        feature_num = [0.15, 0.25, 0.40,
                       0.5, 0.65, 0.75,
                       0.85, 1.0]
        t_p = np.zeros(len(feature_num), dtype=int)
        f_p = np.zeros(len(feature_num), dtype=int)
        t_n = np.zeros(len(feature_num), dtype=int)
        f_n = np.zeros(len(feature_num), dtype=int)
        print(t_p)
        logfile = open(self.data_path + self.file + '_' + self.intent + '_' + str(des) + '_' + str(sampling_index) + '_com_txt_fs_' + str(
                alpha) + '_log.txt', 'w')

        for k in range(fold):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            print("Before FS", X_train.shape, X_test.shape)
            from src.aggregate.feature_selection import FeatureSelector
            feature_selector = FeatureSelector(selection_method=0)
            feature_selector.fit(X_train, y_train)

            if sampling_index == 0:
                X_train, y_train = self.under_sampling(X_train, y_train)
            elif sampling_index == 1:
                X_train, y_train = self.over_sampling(X_train, y_train)
            else:
                X_train, y_train = self.smote(X_train, y_train)

            column = 0
            for f_num in feature_num:
                print("Features", f_num)
                X_temp_train = feature_selector.transform(X_train, int(f_num * total_features), alpha)
                X_temp_test = feature_selector.transform(X_test, int(f_num * total_features), alpha)
                print("After FS", X_temp_train.shape, X_temp_test.shape)
                hypo.fit(X_temp_train, y_train)
                y_predict = hypo.predict(X_temp_test)
                temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
                t_p[column] += temp_tp
                t_n[column] += temp_tn
                f_p[column] += temp_fp
                f_n[column] += temp_fn
                column += 1
            # break

        for x in range(len(feature_num)):
            print(feature_num[x])
            logfile.write(str(feature_num[x])+','+str(int(feature_num[x] * total_features)) + "\n")
            print(t_p[x], t_n[x], f_p[x], f_n[x])
            logfile.write(str(t_p[x]) + "," + str(t_n[x]) + "," + str(f_p[x]) + "," + str(f_n[x]) + "\n")
            acc, pre , rec = self.calc_acc_pre_rec({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]})
            print(self.calc_acc_pre_rec({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]}))
            logfile.write(str(acc) + "," + str(pre) + "," + str(rec)  + "\n")
            fpr, tpr = self.calc_fpr_tpr({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]})
            print(self.calc_fpr_tpr({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]}))
            logfile.write(str(fpr) + "," + str(tpr) + "\n")

        logfile.close()
        return

    # @imbalance @sampling @ensemble @probability @text
    def do_experiment_txt_sampling_ensemble_probability_voting(self, sampling_index: int, hypos:list):
        self.load_data()
        print(self.X_txt.shape)
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y_txt, 10)
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
    # @imbalance @sampling @ensemble @voting @text
    def do_experiment_txt_sampling_ensemble_voting(self, sampling_index: int, hypos:list):
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
        voting_file = open('voting/'+self.file+'_'+self.intent+'_'+ str(sampling_index)+ '.csv', 'w')
        voting_file.write('vote0,vote1,test\n')

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
                output = str(zeros) + ',' + str(ones)+ "," + str(y_test[r])
                print(output)
                voting_file.write(output+'\n')
        voting_file.close()
        return

    # @imbalance @sampling @ensemble @stacking @text
    def do_experiment_txt_sampling_ensemble_stacking(self, sampling_index: int, Hypo, hypos:list):
        self.load_data()
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
        len_hypos = len(hypos)
        stacking_file = open(self.data_path+'stacking/' + self.file + '_' + self.intent + '_' + str(sampling_index) + '.csv', 'w')
        stacking_file.write('prob0,prob1,test\n')
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
            y_predicts_proba_train = np.zeros([len(X_train), 2 * len_hypos], dtype=float)
            # print(y_predicts_proba.shape)
            row = 0
            for k in range(fold):
                X_train_2 = list(X_folds_2)
                X_test_2 = X_train_2.pop(k)
                X_train_2 = np.concatenate(X_train_2)
                y_train_2 = list(y_folds_2)
                y_test_2 = y_train_2.pop(k)
                y_train_2 = np.concatenate(y_train_2)
                if sampling_index== 0:
                    X_train_2, y_train_2 = self.under_sampling(X_train_2, y_train_2)
                elif sampling_index == 1:
                    X_train_2, y_train_2 = self.over_sampling(X_train_2, y_train_2)
                else:
                    X_train_2, y_train_2 = self.smote(X_train_2, y_train_2)

                column = 0
                for hypo in hypos:
                    hypo.fit(X_train_2, y_train_2)
                    y_predict_proba = hypo.predict_proba(X_test_2)
                    for x in range(len(y_predict_proba)):
                        y_predicts_proba_train[row + x][2*column+0] = y_predict_proba[x][0]
                        y_predicts_proba_train[row + x][2*column+1] = y_predict_proba[x][1]
                    column += 1

                row += len(X_test_2)

            y_predicts_proba_test = np.empty([len(X_test), 2 * len_hypos], dtype=float)
            column = 0
            for hypo in hypos:
                hypo.fit(X_train, y_train)
                y_predict_proba = hypo.predict_proba(X_test)
                for x in range(len(y_predict_proba)):
                    y_predicts_proba_test[x][2 * column + 0] = y_predict_proba[x][0]
                    y_predicts_proba_test[x][2 * column + 1] = y_predict_proba[x][1]
                column += 1

            Hypo.fit(y_predicts_proba_train, y_train)
            y_predict_proba = Hypo.predict_proba(y_predicts_proba_test)
            for row in range(len(y_predict_proba)):
                output = str(y_predict_proba[row][0]) + ',' + str(y_predict_proba[row][1])+ "," + str(y_test[row])
                print(output)
                stacking_file.write(output+'\n')

        return


    # @text @str
    def do_experiment_txt_str(self, hypo):
        # classify bug report through text using feature selection
        # classify the structures
        self.load_data()
        print(self.X_str)

        X_str_folds = np.array_split(self.X_str, 10)
        y_folds = np.array_split(self.y_str, 10)
        from src.aggregate.feature_selection import FeatureSelector
        self.X_txt = FeatureSelector().fit_transform_odd_ratio(self.X_txt, self.y_txt, 400, 0.5)

        from sklearn.feature_selection import SelectFdr, chi2
        ch2 = SelectFdr(score_func=chi2, alpha=0.01)
        self.X_str = ch2.fit_transform(self.X_str, self.y)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        print(Counter(self.y_txt))
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
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
            hypo.fit(X_str_train, y_train)
            y_txt_predict_prob = hypo.predict_proba(X_str_train)

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

    # @text @str @weka
    def do_experiment_first_txt_second_categorical_weka(self, hypo1=MultinomialNB(), des:bool=False):
        self.load_data(des=des)
        print(self.X_txt.shape)
        print(self.X_str.shape)
        X_folds = np.array_split(self.X_txt, 10)
        X_str_folds = np.array_split(self.X_str, 10)
        y_folds = np.array_split(self.y_txt, 10)
        for l in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(l)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(l)
            y_train = np.concatenate(y_train)

            X_str_train = list(X_str_folds)
            X_str_test = X_str_train.pop(l)
            X_str_train = np.concatenate(X_str_train)

            fold = 10
            X_folds_2 = np.array_split(X_train, fold)
            y_folds_2 = np.array_split(y_train, fold)

            ## number of second train folds l
            '''
                Generating the test results
                and stores in the y_predicts_prob
            '''

            y_predicts_proba_train = np.zeros([len(X_train), 2], dtype=float)
            y_predicts_proba_test = np.zeros([len(X_test), 2], dtype=float)
            # print(y_predicts_proba.shape)
            row = 0
            for k in range(fold):
                X_train_2 = list(X_folds_2)
                X_test_2 = X_train_2.pop(k)
                X_train_2 = np.concatenate(X_train_2)
                y_train_2 = list(y_folds_2)
                y_test_2 = y_train_2.pop(k)
                y_train_2 = np.concatenate(y_train_2)
                '''---------- Sampling Starts ---------------'''
                X_train_2, y_train_2 = self.under_sampling(X_train_2, y_train_2)
                '''---------- Sampling Ends  ---------------'''
                hypo1.fit(X_train_2, y_train_2)

                y_predict_proba = hypo1.predict_proba(X_test_2)
                for x in range(len(y_predict_proba)):
                    y_predicts_proba_train[row + x][0] = round(y_predict_proba[x][0], 3)
                    y_predicts_proba_train[row + x][1] = round(y_predict_proba[x][1], 3)

                y_predict_proba = hypo1.predict_proba(X_test)
                for x in range(len(y_predict_proba)):
                    y_predicts_proba_test[x][0] = round((y_predicts_proba_test[x][0]*k+y_predict_proba[x][0])/(k+1),3)
                    y_predicts_proba_test[x][1] = round((y_predicts_proba_test[x][1]*k+y_predict_proba[x][1])/(k+1),3)

                row += len(X_test_2)

            '''
                     training with first train data probabilities
            '''
            print(X_str_train.shape, y_predicts_proba_train.shape)
            train_data = np.concatenate((X_str_train, y_predicts_proba_train), axis=1)
            test_data = np.concatenate((X_str_test, y_predicts_proba_test), axis=1)
            '''------------Sampling---------------'''
            train_data, y_train = self.under_sampling(train_data, y_train)
            print("Data", len(train_data), ':', len(test_data))
            '''-----------Sampling---------------------'''
            train_data = np.array(train_data, dtype=float)
            test_data = np.array(test_data, dtype=float)
            print("train_data")
            weka_train_scvfile = open(self.data_path + 'weka/'+self.file+'/'+str(l)+'_'+self.intent+'_'+str(des)+'_train_str.csv', 'w')
            cols = ''
            for i in range(len(self.str_features)):
                cols += str(self.str_features[i]) + ","
            cols += 'prob0,prob1,target'

            print(cols)
            weka_train_scvfile.write(cols+"\n")

            for row in range(len(train_data)):
                output = ''
                for col in range(len(train_data[0])):
                    output += str(round(float(train_data[row][col]),3))+","

                output += str(y_train[row])
                print(output)
                weka_train_scvfile.write(output+"\n")

            weka_train_scvfile.close()
            print("test_data")
            weka_test_scvfile = open(self.data_path + 'weka/' + self.file+'/'+str(l)+'_'+self.intent+'_'+str(des)+'_test_str.csv', 'w')
            cols = ''

            for i in range(len(self.str_features)):
                cols += str(self.str_features[i]) + ","
            cols += 'prob0,prob1,target'

            print(cols)
            weka_test_scvfile.write(cols + "\n")

            for row in range(len(test_data)):
                output = ''
                for col in range(len(test_data[0])):
                    output += str(round(float(test_data[row][col]), 3))+","

                output += str(y_test[row])
                print(output)
                weka_test_scvfile.write(output + "\n")

            weka_test_scvfile.close()

        return
    # @categorical @sampling
    def do_experiment_categorical_data(self, sampling_index, hypo):
        self.load_data()
        print(self.categorical_data.shape)
        print(self.categorical_data)

        # for row in range(len(self.categorical_data)):
        #     output = ''
        #     for col in range(len(self.categorical_data[row])):
        #         output += str(self.categorical_data[row][col])+' '
        #     print(output)
        #
        # exit(4444)

        X_cat_folds = np.array_split(self.categorical_data, 10)
        y_folds = np.array_split(self.y, 10)
        # exit(404)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        print(Counter(self.y))

        for l in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_cat_train = list(X_cat_folds)
            X_cat_test = X_cat_train.pop(l)
            X_cat_train = np.concatenate(X_cat_train)
            y_train = list(y_folds)
            y_test = y_train.pop(l)
            y_train = np.concatenate(y_train)
            if sampling_index == 0:
                X_s, y_s = self.under_sampling(X_cat_train, y_train)
                hypo.fit(X_s, y_s)
            elif sampling_index == 1:
                X_s, y_s = self.over_sampling(X_cat_train, y_train)
                hypo.fit(X_s, y_s)
            else:
                X_s, y_s = self.smote(X_cat_train, y_train)
                hypo.fit(X_s, y_s)

            y_predict = hypo.predict(X_cat_test)
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))

            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return
    # @text @chi2
    def do_experiment_txt_sampling_chi2(self, sampling_index:int, hypo):
        self.load_data()
        print(self.X_txt.shape)
        # print(self.X_txt.shape)
        print(Counter(self.y))
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
        t_p = 0
        f_p = 0
        t_n = 0
        f_n = 0
        logfile = open('feature_select/'+self.file+'_'+self.intent+'_'+str(sampling_index)+'_'+str('chi2')+'_log.txt','w')
        print(self.intent+'_'+str(sampling_index)+'_'+str('chi2'))
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)


            print("Before FS", X_train.shape, X_test.shape)
            from src.aggregate.chi2 import Chi2Selector
            chi_selector = Chi2Selector()
            chi_selector.fit(X_train, y_train)
            critical_value_at_5 = 3.841
            critical_value_at_1 = 2.706
            logfile.write("CHI2\n")
            logfile.write(str(X_train.shape) + " " + str(X_test.shape) + "\n")
            X_train = chi_selector.transform(X_train, critical_value_at_5)
            X_test = chi_selector.transform(X_test, critical_value_at_5)
            logfile.write(str(X_train.shape) + " " + str(X_test.shape) + "\n")

            if sampling_index == 0:
                logfile.write("UnderSampling\n")
                logfile.write(str(X_train.shape)+" "+str(X_test.shape)+"\n")
                X_train, y_train = self.under_sampling(X_train, y_train)
                logfile.write(str(X_train.shape)+" "+str(X_test.shape)+"\n")
            elif sampling_index == 1:
                logfile.write("Oversampling\n")
                logfile.write(str(X_train.shape)+" "+str(X_test.shape)+"\n")
                X_train, y_train = self.under_sampling(X_train, y_train)
                logfile.write(str(X_train.shape)+" "+str(X_test.shape)+"\n")
            elif sampling_index == 2:
                logfile.write("SMOTE\n")
                logfile.write(str(X_train.shape)+" "+str(X_test.shape)+"\n")
                X_train, y_train = self.smote(X_train, y_train)
                logfile.write(str(X_train.shape)+" "+str(X_test.shape)+"\n")


            hypo.fit(X_train, y_train)
            y_predict = hypo.predict(X_test)
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        logfile.write(str(t_p) + "," + str(t_n) + "," + str(f_p) + "," + str(f_n) + " " + "\n")
        acc, pre, rec = self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n})
        print(acc, pre, rec)
        logfile.write(str(acc) + " " + str(pre) + " " + str(rec) + "\n")
        logfile.close()

        return
    # @text @mi
    def do_experiment_txt_sampling_mi(self, sampling_index:int, hypo):
        self.load_data()
        print(self.X_txt)
        print(self.X_txt.shape)
        total_data, total_features = self.X_txt.shape
        X_folds = np.array_split(self.X_txt, 10)
        self.X_txt = []
        y_folds = np.array_split(self.y, 10)
        print(Counter(self.y))
        self.y = []
        print("DIM", total_data, total_features)
        feature_num = [0.15, 0.25, 0.40,
                       0.5, 0.65, 0.75,
                       0.85, 1.0]
        features = np.zeros(len(feature_num), dtype=int)
        t_p = np.zeros(len(feature_num), dtype=int)
        f_p = np.zeros(len(feature_num), dtype=int)
        t_n = np.zeros(len(feature_num), dtype=int)
        f_n = np.zeros(len(feature_num), dtype=int)
        print(t_p)
        logfile = open('feature_select/'+self.file + '_' + self.intent + '_' + str(sampling_index) + '_mi_log.txt', 'w')
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            print("Before FS", X_train.shape, X_test.shape)
            from src.aggregate.mutual_info import MutualInformationSelector
            mutu_info = MutualInformationSelector()
            mutu_info.fit(X_train, y_train)

            if sampling_index == 0:
                X_train, y_train = self.under_sampling(X_train, y_train)
            elif sampling_index == 1:
                X_train, y_train = self.over_sampling(X_train, y_train)
            else:
                X_train, y_train = self.smote(X_train, y_train)

            column = 0
            for f_num in feature_num:
                print("Features", f_num)
                X_temp_train = mutu_info.transform(X_train, int(f_num* total_features))
                X_temp_test = mutu_info.transform(X_test, int(f_num* total_features))
                a, features[column] = X_temp_train.shape
                print("After FS", X_temp_train.shape, X_temp_test.shape)
                hypo.fit(X_temp_train, y_train)
                y_predict = hypo.predict(X_temp_test)
                temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
                t_p[column] += temp_tp
                t_n[column] += temp_tn
                f_p[column] += temp_fp
                f_n[column] += temp_fn
                column += 1
            # break

        for x in range(len(feature_num)):
            print(feature_num[x])
            logfile.write(str(feature_num[x]) + "\n")
            logfile.write(str(total_features *features[x]) + "\n")
            print(t_p, t_n, f_p, f_n)
            logfile.write(str(t_p) + "," + str(t_n) + "," + str(f_p) + "," + str(f_n) + " " + "\n")
            acc, pre, rec = self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n})
            print(acc, pre, rec)
            logfile.write(str(acc) + " " + str(pre) + " " + str(rec) + "\n")
        # logfile.write(self.calc_acc_pre_rec({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]})+ "\n")
        logfile.close()
        return


    # @text @chi2 @feature_extraction
    def do_experiment_feature_extraction_chi2(self):
        print("CHI2")
        self.load_data()
        # print(self.str_features)
        print(len(self.str_features))
        print(self.target_feature)
        data = self.X_str
        target = self.y_str
        print(Counter(target))
        # for row in range(len(data)):
        #     print(data[row])
        from src.aggregate.chi2 import Chi2Selector
        from sklearn.feature_selection import chi2
        chi_selector = Chi2Selector()
        chi_selector.fit(data, target)
        custom_scores = chi_selector.scores()
        sklearn_scores, sklearn_pvalues = chi2(data,target)
        for x in range(len(custom_scores)):
            print(self.str_features[x], custom_scores[x], sklearn_scores[x])
        print("Features")
        features = []
        for x in range(len(self.str_features)):
            features.append([self.str_features[x], custom_scores[x]])

        critical_value_at_5 = 3.841
        critical_value_at_1 = 2.706

        for feature in features:
            if feature[1] >= critical_value_at_5:
               print(feature[0], feature[1])
        return

    # @text @mi @feature_extraction
    def do_experiment_feature_extraction_mi(self):
        print("MI")
        self.load_data()
        print(self.str_features)
        print(len(self.str_features))
        data = self.X_str
        target = self.y_str
        print(Counter(target))
        for row in range(len(data)):
            print(data[row])
        from src.aggregate.mutual_info import MutualInformationSelector
        from sklearn.feature_selection.mutual_info_ import mutual_info_classif
        mi_selector = MutualInformationSelector()
        mi_selector.fit(data, target)
        custom_scores = mi_selector.scores()
        sklearn_scores = mutual_info_classif(data,target)
        for x in range(len(custom_scores)):
            print(self.str_features[x],custom_scores[x], sklearn_scores[x])
        print("Features")
        features =[]
        for x in range(len(self.str_features)):
            features.append([self.str_features[x],custom_scores[x]])
        features = sorted(features, key= lambda score: score[1], reverse=True )
        for feature in features:
            print(feature[0],feature[1])

        return


    # @text @stacking
    def do_experiment_stacking_csv(self, sampling_index=0):
        print(self.file,self.intent)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        alphas = []
        for x in range(10):
            alphas.append(0.35+x*0.05)
        print(alphas)
        y_test=[]
        y_predict=np.zeros([1000,len(alphas)],dtype=int)
        csvfile = open('stacking/'+self.file+'_'+self.intent+'_'+str(sampling_index)+'.csv', newline='')
        reader = csv.DictReader(csvfile)
        row_count = 0
        for row in reader:
            prob0 = float(row['prob0'])
            prob1 = float(row['prob1'])

            test = int(row['test'])
            print(prob0, prob1, test)
            y_test.append(test)
            for x in range(len(alphas)):
                if prob1 >= alphas[x]:
                    y_predict[row_count][x] = 1
                else:
                    y_predict[row_count][x] = 0
            row_count +=1
        csvfile.close()
        print(y_predict)
        for x in range(len(alphas)):
            print(alphas[x])
            t_p, t_n, f_p, f_n = self.calc_tuple(self.confusion_matrix(y_test, y_predict[:,x]))
            print(t_p, t_n, f_p, f_n)
            print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return
    # @text @voting
    def do_experiment_voting_csv(self, sampling_index=0):
        print(self.file,self.intent)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        y_test=[]
        y_predict=np.zeros(1000,dtype=int)
        csvfile = open('voting/'+self.file+'_'+self.intent+'_'+str(sampling_index)+'.csv', newline='')
        reader = csv.DictReader(csvfile)
        row_count = 0
        for row in reader:
            vote0 = float(row['vote0'])
            vote1 = float(row['vote1'])

            test = int(row['test'])
            print(vote0, vote1, test)
            y_test.append(test)
            if vote1 >= vote0:
                y_predict[row_count] = 1
            row_count +=1
        csvfile.close()
        print(y_predict)
        print(Counter(y_test))
        t_p, t_n, f_p, f_n = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return
    # @text @lor
    def do_experiment_txt_sampling_lor(self, sampling_index=0, negation=False, hypo=MultinomialNB()):
        self.load_data()
        print(self.X_txt)
        print(self.X_txt.shape)
        total_data, total_features = self.X_txt.shape
        X_folds = np.array_split(self.X_txt, 10)
        self.X_txt = []
        y_folds = np.array_split(self.y, 10)
        print(Counter(self.y))
        self.y = []
        t_p = 0
        f_p = 0
        t_n = 0
        f_n = 0
        print(t_p)
        logfile = open('feature_select/' + self.file + '_' + self.intent + '_' + str(sampling_index) + '_lor_'+str(negation)+'_log.txt', 'w')
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            from src.aggregate.lor import LORSelector
            lor_selector = LORSelector()
            lor_selector.fit(X_train, y_train)

            if sampling_index == 0:
                X_train, y_train = self.under_sampling(X_train, y_train)
            elif sampling_index == 1:
                X_train, y_train = self.over_sampling(X_train, y_train)
            else:
                X_train, y_train = self.smote(X_train, y_train)

            print("Before FS", X_train.shape, X_test.shape)
            X_temp_train = lor_selector.transform(X_train,  negation=negation)
            X_temp_test = lor_selector.transform(X_test,  negation=negation)
            print("After FS", X_temp_train.shape, X_temp_test.shape)
            hypo.fit(X_temp_train, y_train)
            y_predict = hypo.predict(X_temp_test)
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn
            # break

        print(t_p, t_n, f_p, f_n)
        logfile.write(str(t_p) + "," + str(t_n) + "," + str(f_p) + "," + str(f_n) + "\n")
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        # logfile.write(self.calc_acc_pre_rec({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]})+ "\n")
        logfile.close()
        return
    # @text @features
    def do_experiment_features_lor(self):
        self.load_data()
        print(self.X_txt)
        print(self.X_txt.shape)
        print(Counter(self.y))
        logfile = open('feature_select/' + self.file + '_' + self.intent + '_' + '_lor_feature_score_log.txt', 'w')
        from src.aggregate.lor import LORSelector
        lor_selector = LORSelector()
        lor_selector.fit(self.X_txt, self.y)
        features_scores = lor_selector.scored_features(self.txt_features)
        for feature_score in features_scores:
            print(feature_score)
            logfile.write(str(feature_score[0])+','+str(feature_score[1])+'\n')
        logfile.close()
        return