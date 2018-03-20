import csv,sys, math
from nltk import FreqDist
from nltk.tokenize import regexp_tokenize


class VectorRepresenter:
    def __init__(self, file):
        self.file = file

    # Return the frequency distribution of terms in a text
    def term_count(self,t):
        summary = regexp_tokenize(t, pattern='[a-zA-Z]+')
        proc_t = FreqDist()
        for w in summary:
            proc_t[w] += 1
        return proc_t.most_common()

    # all the terms that will be used as feature
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

    # represent each bug report as vector of terms
    # weight of each term is calculated using tf_idf
    def proc_bug_reports(self, word2vec: bool):
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
            pos = float(row['Security_pos_col'])
            neu = float(row['Security_neu_col'])
            neg = float(row['Security_neg_col'])
            pos = pos / (pos+neu+neg)
            neu = neu / (pos + neu + neg)
            neg = neg / (pos+neu+neg)
            output += str(pos) + ","
            output += str(neu) + ","
            output += str(neg) + ","
            pos = float(row['Performance_pos_col'])
            neu = float(row['Performance_neu_col'])
            neg = float(row['Performance_neg_col'])
            pos = pos / (pos + neu + neg)
            neu = neu / (pos + neu + neg)
            neg = neg / (pos + neu + neg)
            output += str(pos) + ","
            output += str(neu) + ","
            output += str(neg) + ","
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
                    #  word2vec add
                    #  '''
                    if word2vec == True:
                        from gensim.models import KeyedVectors
                        import numpy as np
                        wv_file = self.file
                        if self.file == 'Camel_Shaon':
                            wv_file = 'camel'
                        words_vectors = KeyedVectors.load_word2vec_format(wv_file + '_wv.txt', binary=False)
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
            print(output)
        csvfile.close()
        return

    def vec_process(self, word2vec: bool= False):
        if word2vec == True:
           sys.stdout = open(self.file + 'wv_vec.csv', 'w')
        else:
            sys.stdout = open(self.file + '_vec.csv', 'w')
        self.proc_bug_reports(word2vec)
        sys.stdout.close()
        return