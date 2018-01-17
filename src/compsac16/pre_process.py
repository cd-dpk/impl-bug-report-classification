from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
import csv
import sys,re
import stringcase


def pre_proc_text(t):
    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    t = re.sub('[_]',' ', t)
    ## camel case and Pascal Case splitted
    t =re.sub('(?<=[A-Z])(?=[A-Z][a-z])|(?<=[^A-Z])(?=[A-Z])|(?<=[A-Za-z])(?=[^A-Za-z])',' ',t)
    t = regexp_tokenize(t, pattern='[a-zA-Z_]+')
    processed_text = ''
    for w in t:
        if w not in stopWords:
            w = stemmer.stem(w)
            processed_text = processed_text+' '+stringcase.lowercase(w)
    return processed_text



def pre_proc_text_with_count(t):
    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    t = re.sub('[_]',' ', t)
    ## camel case and Pascal Case splitted
    t =re.sub('(?<=[A-Z])(?=[A-Z][a-z])|(?<=[^A-Z])(?=[A-Z])|(?<=[A-Za-z])(?=[^A-Za-z])',' ',t)
    summary = regexp_tokenize(t, pattern='[a-zA-Z_]+')
    proc_t= FreqDist()
    for w in summary:
        if w not in stopWords:
            w = stemmer.stem(w)
            proc_t[w.lower()] += 1
    return proc_t

def getAllWords(file):
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        word_list =[]
        for row in reader:
            text = row['summary']+" "+row['description']
            dic = pre_proc_text(text)
            terms = dic.most_common()
            for term in dic:
                if term not in word_list:
                    word_list.append(term)
        return word_list





def proc_sum_desc(file):
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        print('issue_id,summary,description,summary_proc,description_proc,Surprising,Dormant,Blocker,Security,Performance,Breakage')
        for row in reader:
            output = str(row['issue_id'] in (None, '') and '' or row['issue_id']) + ','
            output += row['summary'] in (None,'') and '' or row['summary']+','
            output += row['summary'] in (None,'') and '' or pre_proc_text(row['summary'])+','
            output += row['description'] in (None,'') and '' or row['description']+','
            output += row['description'] in (None, '') and '' or pre_proc_text(row['description'])+','
            output += row['Surprising'] in (None, '') and '' or row['Surprising'] + ","
            output += row['Dormant'] in (None, '') and '' or row['Dormant'] + ","
            output += row['Blocker'] in (None, '') and '' or row['Blocker'] + ","
            output += row['Security'] in (None, '') and '' or row['Security'] + ","
            output += row['Performance'] in (None, '') and '' or row['Performance'] + ","
            output += row['Breakage'] in (None, '') and '' or row['Breakage'] + ","
            print(output)


def proc_sum_desc_vec(file):
    word_list= getAllWords(file)
# print(word_list)
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
            dic = pre_proc_text(text)
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
def process():
    file = 'compsac_16.csv'
    sys.stdout= open('compsac_16_proc.csv','w')
    proc_sum_desc(file)
    sys.stdout.close()
