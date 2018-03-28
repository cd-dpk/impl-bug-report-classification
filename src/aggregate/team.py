import sys, csv, re, stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from src.aggregate.grep import GREP
from src.aggregate.pre_processor import TextPreprocessor

class Team:

    def __init__(self, file):
        self.file = file

    # process csv file
    def set_team(self):
        sys.stdout = open(self.file+'_team.csv', 'w', encoding="UTF-8")
        with open('../data/' + self.file + '.csv', newline='', encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile)
            print('team')
            for row in reader:
                assignee = str(row['assignee'] in (None, '') and '' or row['assignee'])
                print(assignee)
        sys.stdout.close()
        return