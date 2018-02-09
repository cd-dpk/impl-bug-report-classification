## learner
import csv,sys, math
from nltk import FreqDist
from nltk.tokenize import regexp_tokenize

from src.aggregate.pre_processor import TextPreprocessor

ambari = 'ambari'
camel = 'camel'
derby = 'derby'
wicket = 'wicket'
all = 'all'
subject =''
Surprising = 'Surprising'
Security = 'Security'
Performance = 'Performance'
intent = ''

def term_count(t):
    summary = regexp_tokenize(t, pattern='[a-zA-Z]+')
    proc_t= FreqDist()
    for w in summary:
        proc_t[w.lower()] += 1
    return proc_t.most_common()

def get_all_terms(file):
    csvfile = open(file+'_proc.csv', newline='')
    reader = csv.DictReader(csvfile)
    word_list = []
    word_df = []
    t_d = 0
    for row in reader:
        processor = TextPreprocessor()
        # text = row['summary'] in (None,'') and '' or row['summary']+" "+row['description'] in (None,'') and '' or row['description']
        text = row['summary'] in (None, '') and '' or row['summary']
        terms = term_count(processor.getProcessedText(text))
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
    return (word_list,word_df, t_d)

def proc_textual_info(file_name):
    word_list, word_df, t_d = get_all_terms(file_name)
    # re_word_list = []
    # re_word_df = []
    # for x in range(len(word_df)):
    #     if word_df[x] >= 2:
    #         re_word_list.append(word_list[x])
    #         re_word_df.append(word_df[x])
    # word_list = re_word_list
    # word_df = re_word_df
    header_str = ''
    header_str += 'reporter,'
    header_str += 'component,'
    header_str += 'keywords,ST,Patch,CE,TC,EN,'
    header_words = ''
    for x in range(len(word_list)):
        header_words += word_list[x] + ','

    # print(header_str + header_words + 'Security')
    print(header_str + header_words + 'Performance')

    csvfile = open(file_name+'_proc.csv', newline='')
    reader = csv.DictReader(csvfile)

    for row in reader:
        output = ''
        output += row['reporter']+","
        output += row['component']+","
        output += row['keywords']+","
        st = str((row['ST'] in (None, '') and '0' or row['ST']))
        patch = str((row['Patch'] in (None, '') and '0' or row['Patch']))
        ce = str((row['CE'] in (None, '') and '0' or row['CE']))
        tc = str((row['TC'] in (None, '') and '0' or row['TC']))
        en = str((row['EN'] in (None, '') and '0' or row['EN']))
        output += st+','+patch+','+ce+','+tc+','+en+','
        # text = row['summary']+" "+row['description']
        txt_processor = TextPreprocessor()
        text = row['summary']
        terms = term_count(txt_processor.getProcessedText(text))
        rw = ''
        for x in range(len(word_list)):
            index = -1
            for t in range(len(terms)):
                if word_list[x] == terms[t][0]:
                    index = t
                    break

            if index != -1:
                weight = terms[index][1]

                # including tf_idf
                weight *= math.log((t_d/word_df[x]), 10)

                ## including word2vec starts
                from gensim.models.word2vec import Word2Vec
                from gensim.models import KeyedVectors
                word_vectors = KeyedVectors.load_word2vec_format('f.txt', binary=False)
                if word_list[x] in word_vectors:
                    vecs = word_vectors[word_list[x]]
                    em_value = 0
                    for x in range(len(vecs)):
                        em_value += vecs[x] * vecs[x]
                    em_value = math.sqrt(em_value)
                    weight *= em_value
                ## including word2vec ends

                rw += str(round(weight, 5))+','
            else:
                rw += '0,'
        output += rw

        # security = str((row['Security'] in (None, '') and '0' or row['Security']))
        performance = str((row['Performance'] in (None, '') and '0' or row['Performance']))

        # output += security
        output += performance
        print(output)
    csvfile.close()
    return

def vec_process(file_name):
    sys.stdout = open(file_name+'_vec.csv','w')
    proc_textual_info(file_name)
    sys.stdout.close()
    return

'''Vector Representation Ends here'''

subject = ambari
vec_process(subject)