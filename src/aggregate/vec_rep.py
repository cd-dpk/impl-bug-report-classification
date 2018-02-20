## learner
import csv,sys, math
from nltk import FreqDist
from nltk.tokenize import regexp_tokenize
class VectorRepresenter:
    def __init__(self, file):
        self.file = file

    def term_count(self,t):
        summary = regexp_tokenize(t, pattern='[a-zA-Z]+')
        proc_t = FreqDist()
        for w in summary:
            proc_t[w] += 1
        return proc_t.most_common()


    def get_all_terms(self):
        csvfile = open(self.file + '_proc.csv', newline='')
        reader = csv.DictReader(csvfile)
        word_list = []
        word_df = []
        t_d = 0
        for row in reader:
            text = row['summary_col'] in (None,'') and '' or row['summary_col']+" "+row['description_col'] in (None,'') and '' or row['description_col']
            # text = row['summary_col'] in (None, '') and '' or row['summary_col']
            terms = self.term_count(text)
            for term in terms:
                index = -1
                for x in range(len(word_list)):
                    if term[0] == word_list[x]:
                        word_df[x] += 1
                        index = x
                if index == -1:
                    word_list.append(term[0])
                    word_df.append(1)
            t_d += 1
        csvfile.close()
        return (word_list, word_df, t_d)

    def proc_bug_reports(self):
        word_list, word_df, t_d = self.get_all_terms()
        '''
        re_word_list = []
        re_word_df = []
        for x in range(len(word_df)):
            if word_df[x] >= 2:
                re_word_list.append(word_list[x])
                re_word_df.append(word_df[x])
        word_list = re_word_list
        word_df = re_word_df
        '''
        header_str = ''
        header_str += 'reporter_col,'
        header_str += 'component_col,'
        header_str += 'Security_pos_col,Security_neu_col,Security_neg_col,Performance_pos_col,Performance_neu_col,Performance_neg_col,ST_col,Patch_col,CE_col,TC_col,EN_col,'
        header_words = ''
        for x in range(len(word_list)):
            header_words += word_list[x] + ','

        print(header_str + header_words + 'target_Security,target_Performance')

        csvfile = open(self.file +'_proc.csv', newline='')
        reader = csv.DictReader(csvfile)

        for row in reader:
            output = ''
            output += row['reporter_col'] + ","
            output += row['component_col'] + ","
            output += row['Security_pos_col'] + ","
            output += row['Security_neu_col'] + ","
            output += row['Security_neg_col'] + ","
            output += row['Performance_pos_col'] + ","
            output += row['Performance_neu_col'] + ","
            output += row['Performance_neg_col'] + ","
            st = str((row['ST_col'] in (None, '') and '0' or row['ST_col']))
            patch = str((row['Patch_col'] in (None, '') and '0' or row['Patch_col']))
            ce = str((row['CE_col'] in (None, '') and '0' or row['CE_col']))
            tc = str((row['TC_col'] in (None, '') and '0' or row['TC_col']))
            en = str((row['EN_col'] in (None, '') and '0' or row['EN_col']))
            output += st + ',' + patch + ',' + ce + ',' + tc + ',' + en + ','
            text = row['summary_col']+" "+row['description_col']
            # text = row['summary']
            terms = self.term_count(text)
            rw = ''
            for x in range(len(word_list)):
                index = -1
                for t in range(len(terms)):
                    if word_list[x] == terms[t][0]:
                        index = t
                        break

                if index != -1:
                    weight = float(terms[index][1])
                    weight *= math.log((float(t_d) / float(word_df[x])), 10)
                    rw += str(round(weight, 5)) + ','
                else:
                    rw += '0,'
            output += rw

            security = str((row['target_Security'] in (None, '') and '0' or row['target_Security']))
            output += security+","
            performance = str((row['target_Performance'] in (None, '') and '0' or row['target_Performance']))
            output += performance
            print(output)
        csvfile.close()
        return

    def vec_process(self):
        sys.stdout = open(self.file+ '_vec.csv', 'w')
        self.proc_bug_reports()
        sys.stdout.close()
        return