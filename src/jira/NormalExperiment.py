from builtins import list
from sklearn.naive_bayes import MultinomialNB
from src.jira.experiment import Experiment
from src.aggregate.pre_processor import TextPreprocessor
from collections import Counter
import numpy as np

class NormalExperiment(Experiment):

    def do_experiment_txt_after_feature_selected(self, l, l1_ratio, hypo):
        self.load_data_featured(l, l1_ratio)
        print(self.X_txt)
        print(self.X_txt.shape)
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
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

    def do_experiment_txt_feature_selection(self, l, l1_ratio, selection_method, hypo):
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
        print(Counter(self.y))
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
            feature_selector = FeatureSelector(selection_method=selection_method)

            X_train = feature_selector.fit_transform(X_train, y_train, l, l1_ratio)
            X_test = feature_selector.transform(X_test)
            print("After FS", X_train.shape, X_test.shape)

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

    def do_experiment_txt(self, hypo):
        self.load_data()
        print(self.X_txt)
        print(self.X_txt.shape)
        X_folds = np.array_split(self.X_txt, 10)
        y_folds = np.array_split(self.y, 10)
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

            print("BEFORE FS", X_train.shape, X_test.shape)
            from sklearn.feature_selection import SelectFdr
            from sklearn.feature_selection import chi2
            ch2 = chi2(X_train, y_train)
            selector = SelectFdr(ch2, alpha=0.05).fit(X_train, y_train)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
            print("AFTER FS", X_train.shape, X_test.shape)

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
    def do_experiment_txt_sampling_ensemble_probability_voting(self, sampling_index: int, hypos: list):
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
                    y_predicts[x][2 * column + 0] = y_predict_proba[x][0]
                    y_predicts[x][2 * column + 1] = y_predict_proba[x][1]
                column += 1

            y_predict = np.empty(len(y_test), dtype=int)
            for r in range(len(y_predict)):
                zeros = 0.0
                ones = 0.0
                for h in range(len_of_hypos):
                    zeros += y_predicts[r][2 * h + 0]
                    ones += y_predicts[r][2 * h + 1]

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
    def do_experiment_txt_sampling_ensemble_voting(self, sampling_index: int, hypos: list):
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
                # print(hypo.predict_proba(X_test))
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
    def do_experiment_txt_sampling_ensemble_stacking(self, sampling_index: int, Hypo, hypos: list):
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

                if sampling_index == 0:
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
                        y_predicts_proba[row + x][2 * column + 0] = y_predict_proba[x][0]
                        y_predicts_proba[row + x][2 * column + 1] = y_predict_proba[x][1]

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
                        y_predicts_proba[x][2 * column + 0] = y_predict_proba[x][0]
                        y_predicts_proba[x][2 * column + 1] = y_predict_proba[x][1]
                    column += 1

                y_predict = Hypo.predict(y_predicts_proba)
                # print(y_predicts_proba)
                # y_predict_proba = Hypo.predict_proba(y_predicts_proba)


                print(self.confusion_matrix(y_test, y_predict))
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
        print(self.X_txt)
        print(self.y)
        print(self.X_txt.shape)
        X_folds = np.array_split(self.X_txt, 10)
        self.X_txt=[]
        y_folds = np.array_split(self.y, 10)
        self.y = []
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

            print("BEFORE FS", X_train.shape, X_test.shape)
            from sklearn.feature_selection import SelectFdr
            from sklearn.feature_selection import chi2
            selector = SelectFdr(chi2, 0.05)
            selector.fit(X_train, y_train)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
            print("AFTER FS", X_train.shape, X_test.shape)


            if sampling_index == 0:
                print("Under Sampling")
                X_train, y_train = self.under_sampling(X_train, y_train)
            elif sampling_index == 1:
                print("Over Sampling")
                X_train, y_train = self.over_sampling(X_train, y_train)
            else:
                print("SMOTE")
                X_train, y_train = self.smote(X_train, y_train)

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

    def do_experiment_featured_terms(self,hypo=MultinomialNB()):
        import csv
        csvfile = open('/media/geet/Files/IITDU/MSSE-03/implementation/src/jira/apache_Security_pos_terms.txt',
                       newline='')
        reader = csv.DictReader(csvfile)

        indices = []

        for row in reader:
            # print(row['index'], row['term'], row['score'])
            indices.append(int(row['index']))
        csvfile.close()

        csvfile = open('/media/geet/Files/IITDU/MSSE-03/implementation/src/jira/apache_Security_neg_terms.txt',
                       newline='')
        reader = csv.DictReader(csvfile)

        for row in reader:
            # print(row['index'], row['term'], row['score'])
            indices.append(int(row['index']))

        csvfile.close()

        print(indices)

        self.load_data()
        print(self.X_txt[:, indices])
        return

    def do_experiment_generate_lexicon_terms(self):
        import sys
        self.load_data()
        print(self.X_txt)
        print(self.y)
        print(self.X_txt.shape)
        print(self.term_features)
        print(len(self.term_features))

        from src.jira.feature_selection import FeatureSelector
        pos, neg, neu = FeatureSelector().get_lexicon_terms(self.X_txt, self.y)

        sys.stdout = open(self.file + '_' + self.intent + '_pos_terms.txt', 'w', encoding="UTF-8")
        for term in pos:
            print(str(term[0]) + "," + self.term_features[term[0]] + "," + str(term[1]))
        sys.stdout.close()

        sys.stdout = open(self.file + '_' + self.intent + '_neg_terms.txt', 'w', encoding="UTF-8")
        for term in neg:
            print(str(term[0]) + "," + self.term_features[term[0]] + "," + str(term[1]))
        sys.stdout.close()

        sys.stdout = open(self.file + '_' + self.intent + '_neu_terms.txt', 'w', encoding="UTF-8")
        for term in neu:
            print(str(term[0]) + "," + self.term_features[term[0]] + "," + str(term[1]))
        sys.stdout.close()

        return

