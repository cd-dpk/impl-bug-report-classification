from src.aggregate.experiment import Experiment
import numpy as np
class VotingExperiment(Experiment):
    def do_voting_experiment(self, hypos:list):
        self.load_data()
        self.feature_selection(100)

        X_folds = np.array_split(self.X, 10)
        y_folds = np.array_split(self.y, 10)

        t_p = 0.0
        f_p = 0.0
        t_n = 0.0
        f_n = 0.0
        len_of_hypos = len(hypos)

        for k in range(10):
            # We use 'list' to copy, in order to 'pop' later on
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)

            X_s, y_s = self.smote(X_train,y_train)

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

            # print(self.confusion_matrix(y_test,y_predict))
            temp_tp, temp_tn, temp_fp, temp_fn = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
            t_p += temp_tp
            t_n += temp_tn
            f_p += temp_fp
            f_n += temp_fn

        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return
