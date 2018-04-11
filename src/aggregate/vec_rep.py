import csv,sys, math
from nltk import FreqDist
from nltk.tokenize import regexp_tokenize


class VectorRepresenter:

    def __init__(self, data_path, file):
        self.file = file
        self.data_path = data_path

    # Return the frequency distribution of terms in a text
    def term_count(self, t):
        summary = regexp_tokenize(t, pattern='[a-zA-Z]+')
        proc_t = FreqDist()
        for w in summary:
            proc_t[w] += 1
        return proc_t.most_common()

    # all the terms that will be used as feature
    def get_all_terms(self, des:bool=False):
        csvfile = open(self.data_path + self.file + '_txt_proc.csv', encoding='UTF-8', newline='')
        reader = csv.DictReader(csvfile)
        word_list = []
        word_df = []
        t_d = 0
        for row in reader:
            text = row['summary_col'] in (None,'') and '' or row['summary_col']
            if des:
                text += " " + row['description_col'] in (None, '') and '' or row['description_col']
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

    # represent each bug report as vector of terms
    # weight of each term is calculated using tf_idf
    def proc_bug_reports_str(self, output_file, ):
        str_file = open(self.data_path + output_file, 'w', encoding='UTF-8')
        header_str = 'issue_id,'
        header_str += 'reporter_col,team_col,'
        header_str += 'component_col,'
        header_str += 'grep_sec,grep_perf,ST_col,Patch_col,CE_col,TC_col,EN_col,'
        str_file.write(header_str +'target_Security,target_Performance\n')
        csvfile = open(self.data_path + self.file +'_str_proc.csv', encoding='UTF-8', newline='')
        reader = csv.DictReader(csvfile)
        for row in reader:
            output = row['issue_id'] + ","
            output += row['reporter_col'] + ","
            output += row['team_col'] + ","
            output += row['component_col'] + ","
            output += row['grep_sec'] + ","
            output += row['grep_perf'] + ","
            st = str((row['ST_col'] in (None, '') and '0' or row['ST_col']))
            patch = str((row['Patch_col'] in (None, '') and '0' or row['Patch_col']))
            ce = str((row['CE_col'] in (None, '') and '0' or row['CE_col']))
            tc = str((row['TC_col'] in (None, '') and '0' or row['TC_col']))
            en = str((row['EN_col'] in (None, '') and '0' or row['EN_col']))
            output += st + ',' + patch + ',' + ce + ',' + tc + ',' + en + ','
            security = str((row['target_Security'] in (None, '') and '0' or row['target_Security']))
            output += security+","
            performance = str((row['target_Performance'] in (None, '') and '0' or row['target_Performance']))
            output += performance
            str_file.write(output+'\n')
        csvfile.close()
        str_file.close()
        return

    def proc_bug_reports_txt(self, word2vec: bool, dim: int=0, src:bool=False, des:bool=False):
        output_file = self.data_path + self.file + '_' + str(word2vec) + '_' + str(dim) + '_' + str(src) + '_' + str(des) + '_vec.csv'
        txt_file = open(output_file, 'w', encoding='UTF-8')
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
        header_str = 'issue_id,'
        header_words = ''
        for x in range(len(word_list)):
            header_words += word_list[x] + ','
        txt_file.write(header_str + header_words + 'target_Security,target_Performance\n')

        csvfile = open(self.data_path + self.file +'_txt_proc.csv', encoding='UTF-8', newline='')
        reader = csv.DictReader(csvfile)

        for row in reader:
            output = row['issue_id'] + ","
            text = row['summary_col']
            if des:
                text += " "+row['description_col']

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
                    #  word2vec add
                    #  '''
                    if word2vec == True:
                        from gensim.models import KeyedVectors
                        import numpy as np
                        wv_file = self.file
                        words_vectors = KeyedVectors.load_word2vec_format(self.data_path + wv_file + '_' + str(dim) + '_' + str(src) +'_wv.txt', binary=False)
                        word_vecs = []

                        if word_list[x] in words_vectors:
                            for vec in words_vectors[word_list[x]]:
                                word_vecs.append(vec * weight)
                        else:
                            word_vecs = [0.0]

                        word_vecs = np.array(word_vecs)
                        weight = np.mean(word_vecs)
                    # '''
                    rw += str(round(weight, 5)) + ','
                else:
                    rw += '0,'
            output += rw

            security = str((row['target_Security'] in (None, '') and '0' or row['target_Security']))
            output += security+","
            performance = str((row['target_Performance'] in (None, '') and '0' or row['target_Performance']))
            output += performance
            txt_file.write(output+'\n')
        csvfile.close()
        txt_file.close()
        return

    def vec_process(self, word2vec: bool= False, dim:int=0, src:bool=False, des:bool= False, txt: bool=False, str: bool=False):
        if str:
            self.proc_bug_reports_str(self.file + '_str_vec.csv')
        if txt:
            self.proc_bug_reports_txt(word2vec, dim, src, des)

        return