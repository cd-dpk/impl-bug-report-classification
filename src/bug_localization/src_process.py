import sys,csv,re,stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from src.aggregate.pre_processor import TextPreprocessor


class SrcProcessor:

    def __init__(self, file):
        self.file = file
        # which single character is removed
        self.additionalStopWords = []
        for x in range(26):
            self.additionalStopWords.append(chr(ord('a') + x))
        self.vocabulary = FreqDist()

    def proc_csv_file(self):
        #
        with open("/media/geet/Files/IITDU/MSSE-03/SRC_P/" + self.file + "_term.csv", newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            print('class_id,class_name,class_content')
            for row in reader:
                class_id = str(row['file_no'] in (None, '') and '' or row['file_no'])
                clas_content = row['proc'] in (None, '') and ' ' or row['proc']
                class_name = row['path'] in (None, '') and ' ' or row['path']

                t_p = TextPreprocessor()
                temp_content = ''

                for term in t_p.getProcessedText(clas_content):
                    temp_content += ' ' + term
                clas_content = temp_content
                print(class_id + "," + class_name + "," + clas_content)

        return


    def pre_process_src(self):
        # ambari 1363,
        # derby
        sys.stdout = open(self.file+'_proc.csv','w',encoding="UTF-8")
        self.proc_csv_file()
        # self.proc_xml_file()
        sys.stdout.close()
        return

# SrcProcessor('ambari').pre_process_src()
# SrcProcessor('camel').pre_process_src()
# SrcProcessor('derby').pre_process_src()
# SrcProcessor('wicket').pre_process_src()

