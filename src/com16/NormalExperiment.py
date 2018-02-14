from builtins import list

from src.com16.experiment import Experiment
from collections import Counter
import numpy as np
import os


class NormalExperiment(Experiment):

    def do_voting_experiment_txt(self, hypos: list):
        cur_dir = "."
        files = os.listdir(cur_dir)
        if self.file+'_'+self.intent+'_proc.csv' not in files:
            print("Sorry!,"+self.file+'_'+self.intent+'_proc.csv file not present')
            exit(404)

        if self.file+'_'+self.intent+'_vec.csv' not in files:
            print("Sorry!," + self.file+'_'+self.intent+'_vec.csv file not present')
            exit(404)

        self.load_data()
        from sklearn.feature_selection import chi2
        from sklearn.feature_selection import SelectFdr
        from sklearn.feature_selection import SelectKBest
        ch2 = SelectFdr(score_func=chi2, alpha=0.01)
        print(self.X_txt.shape)
        self.X_txt = ch2.fit_transform(self.X_txt, self.y)
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
            y_predicts = np.empty([len(y_test), len_of_hypos], dtype=int)
            column = 0
            for hypo in hypos:
                hypo.fit(X_s,y_s)
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

                # if y_predict[r] == 1 or y_test[r] == 1:
                #     print(y_predicts[r], ones, zeros, y_predict[r], y_test[r])

            # print(self.confusion_matrix(y_test,y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return


    def do_agg_experiment_txt(self, hypos: list):
        cur_dir = "."
        files = os.listdir(cur_dir)
        if self.file+'_'+self.intent+'_proc.csv' not in files:
            print("Sorry!,"+self.file+'_'+self.intent+'_proc.csv file not present')
            exit(404)

        if self.file+'_'+self.intent+'_vec.csv' not in files:
            print("Sorry!," + self.file+'_'+self.intent+'_vec.csv file not present')
            exit(404)

        self.load_data()
        from sklearn.feature_selection import chi2
        from sklearn.feature_selection import SelectFdr
        from sklearn.feature_selection import SelectKBest
        ch2 = SelectFdr(score_func=chi2, alpha=0.01)
        print(self.X_txt.shape)
        self.X_txt = ch2.fit_transform(self.X_txt, self.y)
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
            y_predicts = np.zeros([len(y_test), 2], dtype=float)

            for hypo in hypos:
                hypo.fit(X_s, y_s)
                # print(hypo.predict_proba(X_test))
                y_predict_prob = hypo.predict_proba(X_test)
                # print(y_predict_prob)

                for x in range(len(X_test)):
                    # print(y_predict_prob[x][0],y_predict_prob[x][1])
                    y_predicts[x][0] += y_predict_prob[x][0]
                    y_predicts[x][1] += y_predict_prob[x][1]

                # print(y_predicts)

            y_predict = np.zeros(len(y_test), dtype=int)

            for r in range(len(y_predict)):
                if y_predicts[r][1] >= y_predicts[r][0]:
                    y_predict[r] = 1
                else:
                    y_predict[r] = 0
                print(y_predicts[r][0],y_predicts[r][1], y_predict[r])

            # print(self.confusion_matrix(y_test,y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return

    def do_voting_experiment_str(self, hypos: list):
        cur_dir = "."
        files = os.listdir(cur_dir)
        if self.file+'_proc.csv' not in files:
            print("Sorry!,"+self.file+'_proc.csv file not present')
            exit(404, 1)

        if self.file+'_vec.csv' not in files:
            print("Sorry!," + self.file + '_proc.csv file not present')
            exit(404, 2)

        self.load_data()
        X_folds = np.array_split(self.X_str, 10)
        y_folds = np.array_split(self.y, 10)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        len_of_hypos = len(hypos)
        print(self.X_str.shape)
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            print(X_train.shape, X_test.shape)

            from sklearn.feature_selection import chi2
            from sklearn.feature_selection import SelectFdr
            ch2 = SelectFdr(chi2, alpha=0.05)
            X_train = ch2.fit_transform(X_train, y_train)
            X_test = ch2.transform(X_test)

            print(X_train.shape, X_test.shape)

            X_s, y_s = self.smote(X_train, y_train)
            y_predicts = np.empty([len(y_test),len_of_hypos], dtype=int)
            column = 0
            for hypo in hypos:
                hypo.fit(X_s,y_s)
                y_predict = hypo.predict(X_test)
                for x in range(len(y_predict)):
                    y_predicts[x][column] = y_predict[x]

                column += 1

            y_predict = np.empty(len(y_test),dtype=int)
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

                if y_predict[r] == 1 or y_test[r] == 1:
                    print(y_predicts[r], ones, zeros, y_predict[r], y_test[r])

            # print(self.confusion_matrix(y_test,y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return

    def do_voting_experiment_txt_str(self, hypos1: list,hypos2:list):
        cur_dir = "."
        files = os.listdir(cur_dir)
        if self.file+'_proc.csv' not in files:
            print("Sorry!,"+self.file+'_proc.csv file not present')
            exit(404, 1)

        if self.file+'_vec.csv' not in files:
            print("Sorry!," + self.file + '_proc.csv file not present')
            exit(404, 2)
        self.load_data()
        X_folds_txt = np.array_split(self.X_txt, 10)
        X_folds_str = np.array_split(self.X_str, 10)
        y_folds = np.array_split(self.y, 10)
        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        len_of_hypos1 = len(hypos1)
        len_of_hypos2 = len(hypos2)
        print(self.X_txt.shape)
        print(self.X_str.shape)
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train_txt = list(X_folds_txt)
            X_test_txt = X_train_txt.pop(k)
            X_train_txt = np.concatenate(X_train_txt)

            X_train_str = list(X_folds_str)
            X_test_str = X_train_str.pop(k)
            X_train_str = np.concatenate(X_train_str)

            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            X_s, y_s = self.smote(X_train_txt, y_train)
            y_predicts = np.empty([len(y_test),len_of_hypos1], dtype=int)
            column = 0
            for hypo in hypos1:
                hypo.fit(X_s,y_s)
                y_predict = hypo.predict(X_test_txt)
                for x in range(len(y_predict)):
                    y_predicts[x][column] = y_predict[x]

                column += 1

            y_predict = np.empty(len(y_test),dtype=int)
            for r in range(len(y_predict)):
                zeros = 0
                ones = 0
                for h in range(len_of_hypos1):
                    if y_predicts[r][h] == 0:
                        zeros += 1
                    elif y_predicts[r][h] == 1:
                        ones += 1
                if ones >= zeros:
                    y_predict[r] = 1
                else:
                    y_predict[r] = 0

                if y_predict[r] == 1 or y_test[r] == 1:
                    print(y_predicts[r], ones, zeros, y_predict[r], y_test[r])

            # print(self.confusion_matrix(y_test,y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return



    def do_stacking_experiment_txt(self, Hypo, hypos: list):
        cur_dir = "."
        files = os.listdir(cur_dir)
        if self.file+'_proc.csv' not in files:
            print("Sorry!,"+self.file+'_proc.csv file not present')
            exit(404, 1)

        if self.file+'_vec.csv' not in files:
            print("Sorry!," + self.file + '_proc.csv file not present')
            exit(404, 2)

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
                and strores in the y_predicts_prob
            
            '''
            y_predicts_proba = np.empty([len(X_train), 2*len_hypos], dtype=int)
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
                        y_predicts_proba[row + x][column] = y_predict_proba[x][0]
                        y_predicts_proba[row + x][column+1] = y_predict_proba[x][1]
                    column += 1
                row += len(y_test_2)

            '''
                Stacking traing with first train data probabilities
            '''

            Hypo.fit(y_predicts_proba, y_train)

            '''
                Predicting first test data with stacking
            '''

            y_predicts_proba = np.empty([len(y_test), 2*len_hypos], dtype=int)

            column = 0
            for hypo in hypos:
                y_predict_proba = hypo.predict_proba(X_test)
                for x in range(len(y_predict_proba)):
                    y_predicts_proba[x][column] = y_predict_proba[x][0]
                    y_predicts_proba[x][column+1] = y_predict_proba[x][1]
                column += 1

            y_predict = Hypo.predict(y_predicts_proba)

            print(self.confusion_matrix(y_test, y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return




    def test(self):
        cur_dir = "."
        files = os.listdir(cur_dir)
        if self.file+'_proc.csv' not in files:
            print("Sorry!,"+self.file+'_proc.csv file not present')
            exit(404, 1)

        if self.file+'_vec.csv' not in files:
            print("Sorry!," + self.file + '_proc.csv file not present')
            exit(404, 2)
        self.load_data()
        from sklearn.feature_selection import chi2
        ch2s, ps = chi2(self.X_str, self.y)
        print(self.X_str.shape)
        print(ch2s)
        print(ps)
        for p in range(len(ps)):
            if ps[p] <= 0.05:
                print(p, ps[p])
            else:
                print("")




    def test_fs(self):
        cur_dir = "."
        files = os.listdir(cur_dir)
        if self.file + '_' + self.intent + '_proc.csv' not in files:
            print("Sorry!," + self.file + '_' + self.intent + '_proc.csv file not present')
            exit(404)

        if self.file + '_' + self.intent + '_vec.csv' not in files:
            print("Sorry!," + self.file + '_' + self.intent + '_vec.csv file not present')
            exit(404)

        self.load_data()
        from src.aggregate.feature_selection import FeatureSelector
        print(self.X_txt)
        print(Counter(self.y))
        X = FeatureSelector().fit_transform_odd_ratio(self.X_txt,self.y,l=500)
        print(X)

        return

    def do_experiment_txt_fs(self, hypo):
        cur_dir = "."
        files = os.listdir(cur_dir)
        if self.file+'_'+self.intent+'_proc.csv' not in files:
            print("Sorry!,"+self.file+'_'+self.intent+'_proc.csv file not present')
            exit(404)

        if self.file+'_'+self.intent+'_vec.csv' not in files:
            print("Sorry!," + self.file+'_'+self.intent+'_vec.csv file not present')
            exit(404)

        self.load_data()

        from src.aggregate.feature_selection import FeatureSelector
        fs = FeatureSelector()
        print(self.X_txt.shape)
        self.X_txt = fs.fit_transform_odd_ratio(self.X_txt, self.y,l=500,l1_ratio=0.45)
        print(self.X_txt.shape)

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
            X_s, y_s = self.smote(X_train, y_train)
            hypo.fit(X_s,y_s)
            y_predict = hypo.predict(X_test)
            # print(self.confusion_matrix(y_test,y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return

    def do_experiment_txt(self, hypo):
        cur_dir = "."
        files = os.listdir(cur_dir)

        if self.file+'_'+self.intent+'_proc.csv' not in files:
            print("Sorry!,"+self.file+'_'+self.intent+'_proc.csv file not present')
            exit(404)

        if self.file+'_'+self.intent+'_vec.csv' not in files:
            print("Sorry!," + self.file+'_'+self.intent+'_vec.csv file not present')
            exit(404)

        self.load_data()
        from sklearn.feature_selection import chi2
        from sklearn.feature_selection import SelectFdr
        from sklearn.feature_selection import SelectKBest
        ch2 = SelectFdr(score_func=chi2, alpha=0.05)
        print(self.X_txt.shape)
        self.X_txt = ch2.fit_transform(self.X_txt, self.y)
        print(self.X_txt.shape)
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
            X_s, y_s = self.smote(X_train, y_train)
            hypo.fit(X_s, y_s)
            y_predict = hypo.predict(X_test)
            # print(self.confusion_matrix(y_test,y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))

        return
