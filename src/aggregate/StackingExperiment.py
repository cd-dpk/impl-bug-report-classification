from src.aggregate.experiment import Experiment
import numpy as np
class StackingExperiment(Experiment):

    def do_stacking_experiment(self, Hypo, hypos: list):
        self.load_data()
        self.feature_selection(100)

        X_reporter_folds = np.array_split(self.chou_data.reporter_to_numeric_data(), 10)
        X_component_folds = np.array_split(self.chou_data.component_to_numeric_data(), 10)
        X_keywords_folds = np.array_split(self.chou_data.keywords_data, 10)

        X_folds = np.array_split(self.X, 10)
        y_folds = np.array_split(self.y, 10)

        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0

        len_hypos = len(hypos)
        counter = 0
        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            X_folds_2 = np.array_split(X_train, 10)
            y_folds_2 = np.array_split(y_train, 10)

            ## number of second train folds l
            ## generating the test results
            y_predicts = np.empty([len(X_train), len_hypos], dtype=int)
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
                for hypo in hypos:
                    hypo.fit(X_s, y_s)
                    y_predict = hypo.predict(X_test_2)
                    for x in range(len(y_predict)):
                        y_predicts[row + x][column] = y_predict[x]
                    column += 1
                row += len(y_test_2)
            ########################################################################################
            ## stacking traing with first train data ###

            Hypo.fit(y_predicts, y_train)

            ## predicting first test data with stacking
            y_predicts = np.empty([len(y_test), len_hypos], dtype=int)
            column = 0
            for hypo in hypos:
                y_predict = hypo.predict(X_test)
                for x in range(len(y_predict)):
                    y_predicts[x][column] = y_predict[x]
                column += 1

            y_predict = Hypo.predict(y_predicts)

            print(self.confusion_matrix(y_test, y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return
