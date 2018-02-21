from src.aggregate.chou_data import ChouDataHandler
import numpy as np


class Experiment:
    def __init__(self, file, intent):
        self.file = file
        self.intent = intent
        self.sampling_indices = [0, 1, 2]

    def load_data(self):
        chou_data = ChouDataHandler(self.file, self.intent)
        chou_data.load_data()
        self.X_txt = chou_data.textual_data
        self.y = chou_data.target_data
        self.X_str = chou_data.get_numeric_str_data()
        component_data = chou_data.component_to_numeric_data()
        reporter_data = chou_data.reporter_to_numeric_data()
        self.categorical_data = np.empty([len(self.X_txt), 9+len(component_data[0])], dtype=object)
        self.categorical_data_features = []
        self.categorical_data_features.append('author')
        self.categorical_data_features.append('pos')
        self.categorical_data_features.append('neu')
        self.categorical_data_features.append('neg')
        for i in range(len(component_data[0])):
            self.categorical_data_features.append('com'+str(i))
        self.categorical_data_features.append('ST')
        self.categorical_data_features.append('Patch')
        self.categorical_data_features.append('CE')
        self.categorical_data_features.append('TC')
        self.categorical_data_features.append('EN')
        for i in range(len(self.X_txt)):
            k = 0
            self.categorical_data[i][k] = str(reporter_data[i])
            k += 1

            for j in range(len(chou_data.lexicon_data[0])):
                self.categorical_data[i][k] = chou_data.lexicon_data[i][j]
                k += 1

            for j in range(len(component_data[0])):
                self.categorical_data[i][k] = component_data[i][j]
                k += 1

            for j in range(len(chou_data.description_data[i])):
                self.categorical_data[i][k] = chou_data.description_data[i][j]
                k += 1

        return

    def under_sampling(self,X,y):
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()
        return rus.fit_sample(X, y)

    def over_sampling(self,X,y):
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler()
        return ros.fit_sample(X, y)

    def smote(self, X, y):
        from imblearn.over_sampling import SMOTE
        sm = SMOTE()
        return sm.fit_sample(X, y)


    def confusion_matrix(self, y_test, y_predict):
        t_p = 0.0
        t_n = 0.0
        f_p = 0.0
        f_n = 0.0
        for i in range(len(y_predict)):
            if y_test[i] == 1:
                if y_predict[i] == 1:
                    t_p += 1.0
                else:
                    f_n += 1
                continue
            if y_test[i] == 0:
                if y_predict[i] == 1:
                    f_p += 1
                else:
                    t_n += 1

        return {'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}


    def calc_tuple(self,result_dic:dict):
        return (result_dic['t_p'], result_dic['t_n'], result_dic['f_p'], result_dic['f_n'])


    def calc_acc_pre_rec(self,result_dic:dict):
        t_p = result_dic['t_p']
        t_n = result_dic['t_n']
        f_p = result_dic['f_p']
        f_n = result_dic['f_n']

        if (t_p + f_p) != 0 and (t_p + f_n) != 0:
            pre = t_p / (t_p + f_p)
            rec = t_p / (t_p + f_n)
            acc = (t_p + t_n) / (t_p + f_p + t_n + f_n)
            return (acc, pre, rec)
        else:
            return (0.0, 0.0, 0.0)


    def calc_test(self,result_dic:dict):
        t_p = result_dic['t_p']
        t_n = result_dic['t_n']
        f_p = result_dic['f_p']
        f_n = result_dic['f_n']
        return (t_p, t_n)
