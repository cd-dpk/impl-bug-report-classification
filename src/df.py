from nltk.tokenize import regexp_tokenize
import csv
import re
import sys
from nltk import FreqDist
file = 'ambari'

sys.stdout = open('data/df_vec.txt','w')

with open("data/proc/" + file + ".csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    counter = 0
    # write header
    with open('data/word.txt', 'r') as txtfile:
        header ='';
        for line in txtfile:
            line = re.sub('\n','',line)
            header +=line+','
        print(header)
    txtfile.close()
    words = {}
    for row in reader:
        counter = counter + 1;
        proc_summary = row['proc_summary'] and '' or row['proc_summary']
        if proc_summary not in (None, ''):
            proc_summary = regexp_tokenize(proc_summary, pattern='\w+')
            proc_summary_freq = FreqDist(proc_summary).most_common()

            counter = 0
            for w in proc_summary_freq:
                words.__setitem__(proc_summary_freq[counter][0],proc_summary_freq[counter][1]+1)
                counter=counter+1
    with open('data/word.txt', 'r') as txtfile:
        rw = '';
        for line in txtfile:
            line = re.sub('\n', '', line)
            if line in words.keys():
                rw += str(words[line])
                rw += ','
            else:
                rw += "0,"
        print(rw)
    txtfile.close()
sys.stdout.close()