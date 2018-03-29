from src.aggregate.chou_data import ChouDataHandler
import numpy as np


class Experiment:
    def __init__(self, data_path, file, intent):
        self.data_path = data_path
        self.file = file
        self.intent = intent
        self.sampling_indices = [0, 1, 2]

    def load_data(self, word2vec: bool= False, dim: int = 0, src: bool= False):
        chou_data = ChouDataHandler(self.data_path, self.file, self.intent)
        chou_data.load_txt_data(word2vec, dim, src)
        chou_data.load_str_data()
        self.str_features = chou_data.str_features
        self.txt_features = chou_data.text_features
        self.X_txt = chou_data.textual_data
        self.y_txt = chou_data.txt_target_data
        self.X_str = chou_data.get_numeric_str_data()
        self.y_str = chou_data.str_target_data

    def under_sampling(self,X,y):
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()
        return rus.fit_sample(X, y)

    def over_sampling(self, X, y):
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

    def calc_fpr_tpr(self,result_dic:dict):
        t_p = result_dic['t_p']
        t_n = result_dic['t_n']
        f_p = result_dic['f_p']
        f_n = result_dic['f_n']

        t_p += 0.001
        t_n += 0.001
        f_p += 0.001
        f_n += 0.001

        fpr = f_p / (f_p+t_n)
        tpr = t_p / (t_p+f_n)

        return (fpr,tpr)

    def calc_test(self,result_dic:dict):
        t_p = result_dic['t_p']
        t_n = result_dic['t_n']
        f_p = result_dic['f_p']
        f_n = result_dic['f_n']
        return (t_p, t_n)

