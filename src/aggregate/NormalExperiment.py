from builtins import list
import re

from setuptools.command.test import test
from sklearn.naive_bayes import MultinomialNB

from src.aggregate.experiment import Experiment
from collections import Counter
import numpy as np
import os, csv

from src.aggregate.feature_selection import FeatureSelector

class NormalExperiment(Experiment):

    # @imbalance @sampling @text @single_classifier
    def do_experiment_txt_sampling_classifier(self, sampling_index:int=0, hypo=MultinomialNB()):
        self.load_data()
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

    # @text @feature selection
    def do_experiment_txt_feature_selection(self, l, l1_ratio, hypo):
        self.load_data()
        print(self.X_txt.shape)
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
        from src.aggregate.feature_selection import FeatureSelector
        self.X_txt = FeatureSelector().fit_transform_odd_ratio(self.X_txt, self.y, l, l1_ratio)
        print(self.X_txt.shape)
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

    # @imbalance @sampling @ensemble @probability @text
    def do_experiment_txt_sampling_ensemble_probability_voting(self, sampling_index: int, hypos:list):
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

    # @imbalance @sampling @ensemble @stacking @text
    def do_experiment_txt_sampling_ensemble_stacking(self, sampling_index: int, Hypo, hypos:list):
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
                        # print(row + x, 2*column+0)
                        # print(row + x, 2*column+1)
                        y_predicts_proba[row + x][2*column+0] = y_predict_proba[x][0]
                        y_predicts_proba[row + x][2*column+1] = y_predict_proba[x][1]

                    column += 1

                row += len(X_test_2)

            '''
                Stacking training with first train data probabilities
            '''
            # print(y_predicts_proba)
            # print(y_predicts_proba.shape)
            Hypo.fit(y_predicts_proba, y_train)

            '''
                Predicting first test data with stacking
            '''

            y_predicts_proba = np.empty([len(X_test), 2 * len_hypos], dtype=float)

            column = 0
            for hypo in hypos:
                y_predict_proba = hypo.predict_proba(X_test)
                for x in range(len(y_predict_proba)):
                    y_predicts_proba[x][2*column+0] = y_predict_proba[x][0]
                    y_predicts_proba[x][2*column+1] = y_predict_proba[x][1]
                column += 1

            # y_predict = Hypo.predict(y_predicts_proba)
            # print(y_predicts_proba)
            y_predict_proba = Hypo.predict_proba(y_predicts_proba)
            import sys
            sys.stdout = open('stacking/' +self.intent+ str(l) + "_" + str(sampling_index) + '_' + self.file + '.csv', 'w')
            print("prob0,prob1,test")
            for row in range(len(y_predict_proba)):
                output = str(y_predict_proba[row][0]) + ',' + str(y_predict_proba[row][1])+ "," + str(y_test[row])
                print(output)
            sys.stdout.close()

            # print(self.confusion_matrix(y_test, y_predict))
            # temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            # t_p += temp_tp
        #     t_n += temp_tn
        #     f_p += temp_fp
        #     f_n += temp_fn
        #
        # print(t_p, t_n, f_p, f_n)
        # print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return

    # @text @str
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

    # @text @str
    def do_experiment_first_txt_second_str(self, hypo1, hypo2):
        self.load_data()
        X_folds = np.array_split(self.X_txt, 10)
        X_str_folds = np.array_split(self.X_str, 10)
        y_folds = np.array_split(self.y, 10)

        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0

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

            y_predicts_proba = np.zeros([len(X_train), 2 * 1], dtype=float)
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

                hypo1.fit(X_train_2, y_train_2)
                y_predict_proba = hypo1.predict_proba(X_test_2)

                for x in range(len(y_predict_proba)):
                    y_predicts_proba[row + x][0] = y_predict_proba[x][0]
                    y_predicts_proba[row + x][1] = y_predict_proba[x][1]


                row += len(X_test_2)

            '''
                 training with first train data probabilities
            '''
            print(y_predicts_proba)
            print(y_predicts_proba.shape)

            print(X_str_train)
            mod_data = np.concatenate((X_str_train, y_predicts_proba), axis=1)
            print(mod_data)
            print(mod_data.shape)
            # mod_data, y_train = self.smote(mod_data,y_train)
            hypo2.fit(mod_data, y_train)

            '''
                Predicting first test data with second learning
            '''

            y_predict_proba = hypo1.predict_proba(X_test)

            mod_data = np.concatenate((X_str_test, y_predict_proba), axis=1)
            y_predict = hypo2.predict(mod_data)

            # print(self.confusion_matrix(y_test, y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return

    def do_experiment_txt_sampling(self, sampling_index=0, hypo=MultinomialNB()):
        self.load_data()
        print(self.X_txt)
        print(self.X_txt.shape)
        X_folds = np.array_split(self.X_txt, 10)
        self.X_txt = []
        y_folds = np.array_split(self.y, 10)
        self.y = []
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        print(t_p)
        print(Counter(self.y))
        logfile = open(self.intent + '_' + str(sampling_index) + '_log.txt', 'w')
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            print("Before FS", X_train.shape, X_test.shape)

            if sampling_index == 0:
                X_train, y_train = self.under_sampling(X_train, y_train)
            elif sampling_index == 1:
                X_train, y_train = self.over_sampling(X_train, y_train)
            else:
                X_train, y_train = self.smote(X_train, y_train)

            print("After FS", X_train.shape, X_test.shape)
            hypo.fit(X_train, y_train)
            y_predict = hypo.predict(X_test)
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


    # @text @str @weka
    def do_experiment_first_txt_second_categorical_weka(self, sampling_index=0,hypo1=MultinomialNB()):
        self.load_data()
        print(self.X_txt.shape)
        print(self.categorical_data.shape)
        X_folds = np.array_split(self.X_txt, 10)
        X_cat_folds = np.array_split(self.categorical_data, 10)
        y_folds = np.array_split(self.y, 10)
        for l in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(l)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(l)
            y_train = np.concatenate(y_train)

            X_cat_train = list(X_cat_folds)
            X_cat_test = X_cat_train.pop(l)
            X_cat_train = np.concatenate(X_cat_train)

            fold = 10
            X_folds_2 = np.array_split(X_train, fold)
            y_folds_2 = np.array_split(y_train, fold)

            ## number of second train folds l
            '''
                Generating the test results
                and stores in the y_predicts_prob
            '''

            y_predicts_proba = np.zeros([len(X_train), 2], dtype=float)
            y_predicts_proba_X_test = np.zeros([len(X_test), 2 * fold], dtype=float)
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
                y_predicts_proba_temp = hypo1.predict_proba(X_test)
                for x in range(len(y_predicts_proba_temp)):
                    y_predicts_proba_X_test[x][2 * k + 0] = y_predicts_proba_temp[x][0]
                    y_predicts_proba_X_test[x][2 * k + 1] = y_predicts_proba_temp[x][1]

                for x in range(len(y_predict_proba)):
                    y_predicts_proba[row + x][0] = round(y_predict_proba[x][0], 2)
                    y_predicts_proba[row + x][1] = round(y_predict_proba[x][1], 2)

                row += len(X_test_2)

            '''
                     training with first train data probabilities
            '''
            # print(y_predicts_proba.shape)
            train_data = np.concatenate((X_cat_train, y_predicts_proba), axis=1)
            y_predict_proba = np.zeros([len(X_test), 2])
            for x in range(len(y_predicts_proba_X_test)):
                    prob_zero = 0.0
                    prob_one = 0.0
                    for y in range(2*fold):
                        if y%2 ==0:
                            prob_zero += y_predicts_proba_X_test[x][y]
                        elif y%2 ==1:
                            prob_one += y_predicts_proba_X_test[x][y]

                    prob_zero /= fold
                    prob_one /= fold
                    y_predict_proba[x][0] = round(prob_zero,3)
                    y_predict_proba[x][1] = round(prob_one,3)
            test_data = np.concatenate((X_cat_test, y_predict_proba), axis=1)
            '''------------Sampling---------------'''
            train_data, y_train = self.under_sampling(train_data, y_train)
            print("Data", len(train_data),':' ,len(test_data))
            '''-----------Sampling---------------------'''
            weka_data = np.concatenate((train_data, test_data), axis=0)
            weka_data = np.array(weka_data, dtype=float)
            target = np.concatenate((y_train, y_test), axis=0)
            # print(Counter(target))
            # print(data.shape)
            # print(target.shape)
            wekascvfile = open('weka/'+str(l)+'_'+self.file+'_'+self.intent+'_str.csv', 'w')
            cols = ''
            for i in range(len(self.str_features)):
                cols += str(self.str_features[i]) + ","
            cols += 'prob0,prob1,target'
            print(cols)
            wekascvfile.write(cols+"\n")
            for row in range(len(weka_data)):
                output = ''
                for col in range(len(weka_data[0])):
                    output += str(weka_data[row][col])+","

                output += str(target[row])
                print(output)
                wekascvfile.write(output+"\n")
            wekascvfile.close()
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

    def do_experiment_txt_sampling_feature_selection(self, sampling_index=0, hypo=MultinomialNB(), alpha=0.5):
        self.load_data()
        print(self.X_txt)
        print(self.X_txt.shape)
        total_data, total_features = self.X_txt.shape
        X_folds = np.array_split(self.X_txt, 10)
        self.X_txt = []
        y_folds = np.array_split(self.y, 10)
        print(Counter(self.y))
        self.y = []
        print("DIM",total_data, total_features)
        feature_num = [int(0.15*total_features), int(0.25*total_features), int(0.40*total_features), int(0.5*total_features),int(0.65* total_features), int(0.75* total_features), int(0.85* total_features), total_features]
        t_p = np.zeros(len(feature_num), dtype=int)
        f_p = np.zeros(len(feature_num), dtype=int)
        t_n = np.zeros(len(feature_num), dtype=int)
        f_n = np.zeros(len(feature_num), dtype=int)
        print(t_p)
        logfile = open(self.file+'_'+self.intent + '_' + str(sampling_index) + '_com_txt_fs_' + str(alpha) + '_log.txt', 'w')
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)
            if sampling_index == 0:
                X_train, y_train = self.under_sampling(X_train, y_train)
            elif sampling_index == 1:
                X_train, y_train = self.over_sampling(X_train, y_train)
            else:
                X_train, y_train = self.smote(X_train, y_train)

            print("Before FS", X_train.shape, X_test.shape)
            from src.aggregate.feature_selection import FeatureSelector
            feature_selector = FeatureSelector(selection_method=0)
            feature_selector.fit(X_train, y_train)
            column = 0
            for f_num in feature_num:
                print("Features", f_num)
                X_temp_train = feature_selector.transform(X_train, f_num, alpha)
                X_temp_test = feature_selector.transform(X_test, f_num, alpha)
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
            print(t_p[x], t_n[x], f_p[x], f_n[x])
            logfile.write(str(t_p[x]) + "," + str(t_n[x]) + "," + str(f_p[x]) + "," + str(f_n[x]) + "\n")
            print(self.calc_acc_pre_rec({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]}))
            # logfile.write(self.calc_acc_pre_rec({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]})+ "\n")
        logfile.close()
        return

    def do_experiment_txt_sampling_chi2(self, sampling_index:int,  hypo):
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
        logfile = open(self.file+'_'+self.intent+'_'+str(sampling_index)+'_'+str('chi2')+'_log.txt','w')
        print(self.intent+'_'+str(sampling_index)+'_'+str('chi2'))
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            X_temp_train = X_train
            y_temp_train = y_train
            X_temp_test = X_test

            if sampling_index == 0:
                logfile.write("UnderSampling\n")
                logfile.write(str(X_temp_train.shape)+" "+str(X_temp_test.shape)+"\n")
                X_temp_train, y_temp_train = self.under_sampling(X_train, y_train)
                logfile.write(str(X_temp_train.shape)+" "+str(X_temp_test.shape)+"\n")
            elif sampling_index == 1:
                logfile.write("Oversampling\n")
                logfile.write(str(X_temp_train.shape)+" "+str(X_temp_test.shape)+"\n")
                X_temp_train, y_temp_train = self.over_sampling(X_train, y_train)
                logfile.write(str(X_temp_train.shape)+" "+str(X_temp_test.shape)+"\n")
            elif sampling_index == 2:
                logfile.write("SMOTE\n")
                logfile.write(str(X_temp_train.shape)+" "+str(X_temp_test.shape)+"\n")
                X_temp_train, y_temp_train = self.smote(X_train, y_train)
                logfile.write(str(X_temp_train.shape)+" "+str(X_temp_test.shape)+"\n")

            from sklearn.feature_selection import SelectFdr
            from sklearn.feature_selection import chi2
            logfile.write("CHI2\n")
            logfile.write(str(X_temp_train.shape) + " " + str(X_temp_test.shape) + "\n")
            selector = SelectFdr(chi2).fit(X_temp_train, y_temp_train)
            X_temp_train = selector.transform(X_temp_train)
            X_temp_test = selector.transform(X_temp_test)
            logfile.write(str(X_temp_train.shape) + " " + str(X_temp_test.shape) + "\n")

            hypo.fit(X_temp_train, y_temp_train)
            y_predict = hypo.predict(X_temp_test)
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
        feature_num = [int(0.15 * total_features), int(0.25 * total_features), int(0.40 * total_features),
                       int(0.5 * total_features), int(0.65 * total_features), int(0.75 * total_features),
                       int(0.85 * total_features), total_features]
        features = np.zeros(len(feature_num), dtype=int)
        t_p = np.zeros(len(feature_num), dtype=int)
        f_p = np.zeros(len(feature_num), dtype=int)
        t_n = np.zeros(len(feature_num), dtype=int)
        f_n = np.zeros(len(feature_num), dtype=int)
        print(t_p)
        logfile = open(
            self.file + '_' + self.intent + '_' + str(sampling_index) + '_mi_log.txt', 'w')
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)
            if sampling_index == 0:
                X_train, y_train = self.under_sampling(X_train, y_train)
            elif sampling_index == 1:
                X_train, y_train = self.over_sampling(X_train, y_train)
            else:
                X_train, y_train = self.smote(X_train, y_train)

            print("Before FS", X_train.shape, X_test.shape)
            from src.aggregate.mutual_info import MutualInformationSelector
            mutu_info = MutualInformationSelector()
            mutu_info.fit(X_train, y_train)
            column = 0
            for f_num in feature_num:
                print("Features", f_num)
                X_temp_train = mutu_info.transform(X_train, f_num)
                X_temp_test = mutu_info.transform(X_test, f_num)
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
            logfile.write(str(features[x]) + "\n")
            print(t_p, t_n, f_p, f_n)
            logfile.write(str(t_p) + "," + str(t_n) + "," + str(f_p) + "," + str(f_n) + " " + "\n")
            acc, pre, rec = self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n})
            print(acc, pre, rec)
            logfile.write(str(acc) + " " + str(pre) + " " + str(rec) + "\n")
        # logfile.write(self.calc_acc_pre_rec({'t_p': t_p[x], 'f_p': f_p[x], 't_n': t_n[x], 'f_n': f_n[x]})+ "\n")
        logfile.close()
        return

    def do_experiment_feature_extraction_chi2(self):
        print("CHI2")
        self.load_data()
        print(self.str_features)
        print(len(self.str_features))
        data = self.X_str
        target = self.y
        print(Counter(target))
        for row in range(len(data)):
            print(data[row])
        from src.aggregate.chi2 import Chi2Selector
        from sklearn.feature_selection import chi2
        chi_selector = Chi2Selector()
        chi_selector.fit(data, target)
        custom_scores = chi_selector.scores()
        sklearn_scores, sklearn_pvalues = chi2(data,target)
        for x in range(len(custom_scores)):
            print(self.str_features[x],custom_scores[x], sklearn_scores[x])
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

    def do_experiment_feature_extraction_mi(self):
        print("MI")
        self.load_data()
        print(self.str_features)
        print(len(self.str_features))
        data = self.X_str
        target = self.y
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