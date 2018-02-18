## learner
import csv, sys, math
from nltk import FreqDist
from nltk.tokenize import regexp_tokenize

class DFRepresenter:
    def __init__(self, file):
        self.file = file

    def term_count(self, t):
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
            text = row['summary'] in (None, '') and '' or row['summary']
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

    def save_idf(self):
        word_list, word_df, t_d = self.get_all_terms()
        print('term', 'idf')
        for i in range(len(word_list)):
            idf = float(t_d)/float(word_df[i])
            idf = math.log(idf, math.e)
            print(str(word_list[i])+","+str(idf))
        return

    def process_idf(self):
        sys.stdout = open(self.file + '_idf.csv', 'w')
        self.save_idf()
        sys.stdout.close()
        return

DFRepresenter('camel').process_idf()
