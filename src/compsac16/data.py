from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
import csv, sys

def pre_proc_text_with_count(t):
    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    summary = regexp_tokenize(t, pattern='[a-zA-Z]+')
    proc_t= FreqDist()
    for w in summary:
        proc_t[w.lower()] += 1
    return proc_t

def getAllWords(file):
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        word_list =[]
        for row in reader:
            text = row['summary_proc'] in (None,'') and '' or row['summary_proc']+" "+row['description_proc'] in (None,'') and '' or row['description_proc']
            dic = pre_proc_text_with_count(text)
            terms = dic.most_common()
            for term in dic:
                if term not in word_list:
                    word_list.append(term)
        return word_list


def proc_sum_desc_vec(file):
    word_list= getAllWords(file)
    #print(word_list)
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        ## include header
        header =''
        for word in word_list:
            header += word+','

        print(header+'Surprising')
        line = 0;
        for row in reader:
            text = row['summary']+" "+row['description']
            dic = pre_proc_text_with_count(text)
            terms = dic.most_common()
            output = ''
            rw=''
            for word in word_list:
                counter = 0
                index = -1
                for t in terms:
                    if word in (terms[counter]) :
                        index = counter
                        break
                if index != -1:
                    rw += str(terms[index][1])+","
                else:
                    rw += '0,'
            output += rw+''
            output += row['Surprising'] in (None, '') and '' or row['Surprising']
            print(output)

def main_process():
    file = 'ambari_proc.csv'
    sys.stdout= open('ambari_data.csv','w')
    proc_sum_desc_vec(file)
    sys.stdout.close()

main_process()