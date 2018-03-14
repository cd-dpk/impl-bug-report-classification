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
        strongSecurityExpression = "(?i)(denial.of.service|\\bXXE\\b|remote.code.execution|\\bopen.redirect|OSVDB|\\bvuln|\\bCVE\\b|\\bXSS\\b|\\bReDoS\\b|\\bN VD\\b|malicious|x−frame−options|attack|cross.site|exploit|directory.traversal|\\bRCE\\b|\\bdos\\b|\\bXSRF\\b|clickjack|session.fixation|hijack|advisory|insecure|security|\\bcross−origin\\b|unauthori[z|s]ed|infinite.loop)";
        mediumSecurityExpression = "(?i)(authenticat(e|ion)|brute force|bypass|constant.time|crack|credential|\\bDoS\\b|expos(e|ing)|hack|harden|injection|lockout|overflow|password|\\bPoC\\b|proof.of.concept|poison|privelage|\\b(in)?secur(e|ity)|(de)?serializ|spoof|timing|traversal)";
        # print(strongSecurityExpression, mediumSecurityExpression)
        text_to_search = summary
        if description:
            text_to_search += ' ' + description
        text_to_search = summary
        grep = strongSecurityExpression+'|'+mediumSecurityExpression
        if re.search(strongSecurityExpression,text_to_search) or re.search(mediumSecurityExpression,text_to_search):
            return (grep,1)
        else:
            return (grep,0)

    def predict_performance_label(self, summary, description):
        perfExpression = "(?i)(\\bCPU\\b|\\bmemory\\b|\\bdisk\\b|\\bperf\\b|\\bperformance\\b|\\bslow(ing)?\\b|response|(times?)|speed|utiliz(e|ing)|call|RAM)"
        # print(perfExpression)
        # perf_kewords = ["perf", "slow", "hang", "performance"]
        text_to_search = summary
        if description:
            text_to_search += ' ' + description
        text_to_search = summary
        grep = perfExpression
        if re.search(perfExpression, text_to_search):
            return (grep,1)
        else:
            return (grep,0)

    def read_and_identify(self, file, intent):
        logfile = open(file+'_'+intent+'_sum_grep.txt','w')
        y_test = []
        y_predict = []
        import xml.etree.ElementTree as ET
        import os
        dir = '/media/geet/Files/IITDU/MSSE-03/DocumentSimilarity-master/Apache/'
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
                        grep, predict_label = self.predict_security_label(summary=summary, description=description)
                        y_predict.append(predict_label)
                        print(id, sec_flag,  predict_label)
                    elif intent == 'Performance':
                        y_test.append(int(perf_flag))
                        grep, predict_label = self.predict_performance_label(summary=summary, description=description)
                        y_predict.append(predict_label)
                        print(id, perf_flag, predict_label)

                    # print(id + "," + key + "," + summary  + "," + sec_flag + "," + perf_flag)
                    # print(seen_ids)
                    seen_ids.append(id)
        print(Counter(y_test))
        print(Counter(y_predict))
        logfile.write(grep+'\n')
        t_p, t_n, f_p, f_n = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
        print(t_p, t_n, f_p, f_n)
        logfile.write(str(t_p)+','+str(t_n)+','+str(f_p)+','+str(f_n)+'\n')
        acc, pre, rec = self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n})
        print(str(acc)+','+str(pre)+','+str(rec)+'\n')
        logfile.write(str(acc)+','+str(pre)+','+str(rec)+'\n')
        logfile.close()
        return

# GREP().read_and_identify('apache','Security')
GREP().read_and_identify('apache','Performance')
# br = 'x−frame-options '
# print(GREP().predict_security_label(br, ''))

