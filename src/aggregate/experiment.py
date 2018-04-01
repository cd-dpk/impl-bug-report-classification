from src.aggregate.chou_data import ChouDataHandler
import numpy as np


class Experiment:
    def __init__(self, data_path, file, intent):
        self.data_path = data_path
        self.file = file
        self.intent = intent
        self.sampling_indices = [0, 1, 2]

    def load_data(self, word2vec: bool= False, dim: int = 0, src: bool= False, des: bool =False):
        chou_data = ChouDataHandler(self.data_path, self.file, self.intent)
        chou_data.load_txt_data(word2vec, dim, src, des=des)
        chou_data.load_str_data()
        chou_data.load_raw_data()
        self.str_features = chou_data.str_features
        self.txt_features = chou_data.text_features
        self.X_txt = chou_data.textual_data
        self.y_txt = chou_data.txt_target_data
        self.X_str = chou_data.get_numeric_str_data()
        self.y_str = chou_data.str_target_data
        self.target_feature = chou_data.target_column
        self.X_raw = chou_data.raw_data
        self.raw_features = chou_data.raw_features
        self.y_raw = chou_data.raw_target

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


    def calc_pre_rec_acc_fpr_tpr(self,result_dic:dict):
        t_p = result_dic['t_p']
        t_n = result_dic['t_n']
        f_p = result_dic['f_p']
        f_n = result_dic['f_n']

        t_p += 0.00001
        t_n += 0.00001
        f_p += 0.00001
        f_n += 0.00001

        p = t_p + f_n
        n = f_p + t_n
        Y = t_p + f_p
        N = f_n + t_n

        pre = t_p / Y
        rec = t_p / p
        acc = (t_p + t_n) / (p+n)
        fpr = f_p / n
        tpr = t_p / p

        return (pre, rec, acc, fpr, tpr)

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

