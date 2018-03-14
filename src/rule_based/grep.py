import re
from collections import Counter
class GREP:

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

    def predict_security_label(self, summary, description):
        return 0;
    def predict_performance_label(self, summary, description):
        return 0

    def read_and_identify(self, file, intent):
        y_test = []
        y_predict = []
        import xml.etree.ElementTree as ET
        import os
        dir = 'C:/Users/Assistant/Dropbox/BugReports/Apache/'
        print('id,key,summary,description,target_Security,target_Performance')
        files = os.listdir(dir)
        seen_ids = []
        for file in files:
            print(file)
            if re.fullmatch(".*com.xml$", file):
                420
            else:
                continue
            # print(dir + file)
            tree = ET.parse(dir + file)
            root = tree.getroot()
            for bug in root.findall('item'):
                id = bug.find('id').text
                if id not in seen_ids:
                    key = bug.find('key').text
                    summary = bug.find('summary').text
                    description = bug.find("description") in (None, '') and '' or bug.find("description").text

                    sec_flag = bug.find("sec").text
                    perf_flag = bug.find("perf").text
                    if intent == 'Security':
                        y_test.append(int(sec_flag))
                        predict_label = self.predict_security_label(summary=summary, description=description)
                        y_predict.append(predict_label)
                        print(id, sec_flag,  predict_label)
                    elif intent == 'Performance':
                        y_test.append(int(perf_flag))
                        predict_label = self.predict_security_label(summary=summary, description=description)
                        y_predict.append(predict_label)
                        print(id, perf_flag, predict_label)

                    # print(id + "," + key + "," + summary  + "," + sec_flag + "," + perf_flag)
                    # print(seen_ids)
                    seen_ids.append(id)
        print(Counter(y_test))
        print(Counter(y_predict))
        t_p, t_n, f_p, f_n = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return

GREP().read_and_identify('apache','Security')
