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
            header+=word+','

        print('issue_id,'+header+'Surprising,Dormant,'
              'Blocker,Security,Performance,Breakage')
        line = 0;
        for row in reader:
            text = row['summary']+" "+row['description']
            dic = pre_proc_text_with_count(text)
            terms = dic.most_common()
            output = str(row['issue_id'] in (None, '') and '' or row['issue_id']) + ''
            rw=''
            for word in word_list:
                counter = 0
                index = -1
                for t in terms:
                    if word in (terms[counter]) :
                        index = counter
                        break
                if index != -1:
                    rw += ','+str(terms[index][1])
                else:
                    rw += ',0'
            output += rw
            output += row['Surprising'] in (None, '') and '' or row['Surprising'] + ","
            output += row['Dormant'] in (None, '') and '' or row['Dormant'] + ","
            output += row['Blocker'] in (None, '') and '' or row['Blocker'] + ","
            output += row['Security'] in (None, '') and '' or row['Security'] + ","
            output += row['Performance'] in (None, '') and '' or row['Performance'] + ","
            output += row['Breakage'] in (None, '') and '' or row['Breakage'] + ","
            line+=1
            print(output)

def main_process():
    file = 'compsac_16_proc.csv'
    sys.stdout= open('compsac_16_vec.csv','w')
    proc_sum_desc_vec(file)
    sys.stdout.close()

main_process()