## learner
import csv, sys, math
from nltk import FreqDist
from nltk.tokenize import regexp_tokenize
class SRCVectorRepresenter:
    def __init__(self, file):
        self.file = file

    def get_all_terms(self):
        csvfile = open(self.file + '_idf.csv', newline='')
        reader = csv.DictReader(csvfile)
        word_list = []
        word_idf = []
        t_d = 0
        for row in reader:
            term = row['term'] in (None, '') and '' or row['term']
            idf = row['idf'] in (None, '') and '' or row['idf']
            word_list.append(term)
            word_idf.append(idf)
            print(term, idf)
        csvfile.close()
        return (word_list, word_idf)

    def proc_src_classes(self):
        word_list, word_idf = self.get_all_terms()
        header_str = ''
        header_str += 'class_id,class_name,'
        header_words = ''
        for x in range(len(word_list)):
            header_words += word_list[x] + ','

        print(header_str + header_words)
        csvfile = open(self.file +'_proc.csv', newline='')
        reader = csv.DictReader(csvfile)

        for row in reader:
            output = ''
            text = row['class_content']
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
                    weight *= float(word_idf[x])
                    rw += str(round(weight, 5)) + ','
                else:
                    rw += '0,'
            output += rw
            print(output)
        csvfile.close()
        return

    def src_vec_process(self):
        # sys.stdout = open(self.file + '_vec.csv', 'w')
        self.proc_src_classes()
        # sys.stdout.close()
        return

SRCVectorRepresenter('camel').src_vec_process()