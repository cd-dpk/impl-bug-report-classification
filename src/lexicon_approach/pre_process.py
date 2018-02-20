import sys,csv,re, stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from src.aggregate.pre_processor import TextPreprocessor


class Preprocessor:

    def __init__(self, file):
        self.file = file


    def proc_csv_file(self):
        with open('../data/' + self.file + '.csv', newline='', encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile)
            print('issue_id,summary,description,target_Security,target_Performance')
            for row in reader:
                issue_id = str(row['issue_id'] in (None, '') and '' or row['issue_id'])
                t_p = TextPreprocessor()
                summary = (row['summary'] in (None, '') and '' or row['summary'])
                temp_summary = ''
                for word in t_p.getProcessedText(summary):
                    temp_summary += ' '+word
                summary = temp_summary
                description = (row['description'] in (None, '') and '' or row['description'])
                temp_description = ''
                for word in t_p.getProcessedText(description):
                    temp_description += ' ' + word
                description = temp_description
                security_label = str((row['Security'] in (None, '') and '0' or row['Security']))
                perf_label = str((row['Performance'] in (None, '') and '0' or row['Performance']))

                print(issue_id + ',' + summary + ',' + description + ',' + security_label + "," + perf_label)

        return


    def pre_process(self):
        sys.stdout = open(self.file+'_lexicon_proc.csv','w',encoding="UTF-8")
        self.proc_csv_file()
        # self.proc_xml_file()
        sys.stdout.close()
        return


