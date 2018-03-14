import sys,csv,re,stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from src.aggregate.pre_processor import TextPreprocessor


class Preprocessor:

    def __init__(self, file, intent):
        self.file = file
        self.intent = intent
        # which single character is removed
        self.additionalStopWords = []
        for x in range(26):
            self.additionalStopWords.append(chr(ord('a') + x))
        self.vocabulary = FreqDist()



    def proc_xml_file(self):
        import xml.etree.ElementTree as ET
        import os
        # sys.stdout = open(self.file + '_' + self.intent + '_proc.csv', 'w', encoding="UTF-8")
        # print('issue_id,summary,description,Security'+self.intent)
        dir = '/media/geet/Files/IITDU/MSSE-03/DocumentSimilarity-master/Apache/'
        # sec_keys = []
        # with open("/media/geet/Files/IITDU/MSSE-03/DocumentSimilarity-master/sec.txt", "r") as myfile:
        #     for line in myfile:
        #         sec_keys.append(re.sub("\n", "", line))
        #
        # print(sec_keys)
        # perf_keys = []
        # with open("/media/geet/Files/IITDU/MSSE-03/DocumentSimilarity-master/perf.txt", "r") as myfile:
        #     for line in myfile:
        #         perf_keys.append(re.sub("\n", "", line))
        # print(perf_keys)
        # print('id,key,summary,proc_summary,target_Security,target_Performance')
        print('id,key,proc_summary,target_Security,target_Performance')
        files = os.listdir(dir)
        sentences = []
        seen_ids = []
        for file in files:
            # print(file)
            if re.fullmatch(".*com.xml$",file):
                420
            else:
                continue
            # print(dir + file)
            tree = ET.parse(dir + file)
            root = tree.getroot()
            for bug in root.findall('item'):
                id = bug.find('id').text
                if id not in seen_ids:
                    key = bug.find('key').text
                    summary = bug.find('summary').text
                    description = bug.find("description").text
                    sec_flag = bug.find("sec").text
                    perf_flag = bug.find("perf").text
                    t_p = TextPreprocessor()

                    proc_summary = ' '
                    for word in t_p.getProcessedText(summary):
                        proc_summary += ' ' + word

                    # proc_description = ' '
                    # for word in t_p.getProcessedText(description):
                    #     proc_description += ' ' + word

                    # print(id,key,summary,proc_summary,description,proc_description,sec_flag,perf_flag)
                    # print(id+","+key+","+summary+","+proc_summary+","+sec_flag+","+perf_flag)
                    print(id + "," + key + "," + proc_summary + "," + sec_flag + "," + perf_flag)
                    # print(seen_ids)
                    seen_ids.append(id)
                # t = TextPreprocessor()
                # line_sentence = []
                # for word in re.split(" ", t.getProcessedText(text=sentence)):
                #     line_sentence.append(word)
                # sentences.append(line_sentence)

        # sys.stdout.close()
        return

    def pre_process(self):
        # sys.stdout = open(self.file+'.csv', 'w', encoding="UTF-8")
        self.proc_xml_file()
        # sys.stdout.close()
        return