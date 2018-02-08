from src.aggregate.experiment import Experiment
import numpy as np

class NormalExperiment(Experiment):

    def doExperiment(self,hypo):
        self.load_data()
        self.feature_selection(100)

        X_folds = np.array_split(self.X, 10)
        y_folds = np.array_split(self.y, 10)

        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        counter = 0
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            X_s, y_s = self.smote(X_train,y_train)

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


    def do_experiment_structural(self, h_f, h_s):
        self.load_data()
        self.feature_selection(100)

        X_folds = np.array_split(self.X, 10)
        y_folds = np.array_split(self.y, 10)

        X_folds_str = np.array_split(self.X_str, 10)

        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0

        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            X_train_str = list(X_folds_str)
            X_test_str = X_train_str.pop(k)
            X_train_str = np.concatenate(X_train_str)

            X_folds_2 = np.array_split(X_train, 10)
            y_folds_2 = np.array_split(y_train, 10)

            y_predicts = np.empty(len(X_train), dtype=int)
            row = 0
            ########################################################################################
            for l in range(10):
                X_train_2 = list(X_folds_2)
                X_test_2 = X_train_2.pop(l)
                X_train_2 = np.concatenate(X_train_2)
                y_train_2 = list(y_folds_2)
                y_test_2 = y_train_2.pop(l)
                y_train_2 = np.concatenate(y_train_2)
                X_s, y_s = self.smote(X_train_2, y_train_2)
                column = 0
                h_f.fit(X_s, y_s)
                y_predict = h_f.predict(X_test_2)
                for x in range(len(y_predict)):
                    y_predicts[row + x] = y_predict[x]

                row += len(y_test_2)
            ########################################################################################

            # print(X_train_str)
            # print(y_predicts)
            X_train_str_m = np.empty([len(X_train_str), len(X_train_str[0]) + 1], dtype=object)
            for x in range(len(y_predicts)):
                X_train_str_m[x] = np.append(X_train_str[x], np.array(y_predicts[x]))

            # print(X_train_str_m)
            X_s,y_s = self.smote(X_train_str_m, y_train)
            h_s.fit(X_s,y_s)

            ## predicting first test data with stacking
            y_predicts = h_f.predict(X_test)

            X_test_str_m = np.empty([len(X_test_str), len(X_test_str[0]) + 1], dtype=object)
            for x in range(len(y_predicts)):
                X_test_str_m[x] = np.append(X_test_str[x], np.array(y_predicts[x]))

            # print(X_test_str_m)

            y_predict = h_s.predict(X_test_str_m)

            # print(self.confusion_matrix(y_test, y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return