import sys,csv,re,stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist

class TextPreprocessor:

    stop_words = set(stopwords.words('english'))
    additional_stop_words = []

    def getProcessedText(self,text):
        stemmer = PorterStemmer()
        text = re.sub('[_]', ' ', text)
        ## camel case and Pascal Case splitted
        text = re.sub('(?<=[A-Z])(?=[A-Z][a-z])|(?<=[^A-Z])(?=[A-Z])|(?<=[A-Za-z])(?=[^A-Za-z])', ' ', text)
        tokens = regexp_tokenize(text, pattern='[a-zA-Z]+')
        # print('Tokens:',tokens)
        processed_text = ''
        for w in tokens:
            w = stringcase.lowercase(w)
            ## when to use percentage of vocabulary
            ## n = int(len(vocabulary)*0.1)
            ## If src vocabulary is used,then
            # if w not in stopWords and w not in additionalStopWords and w not in vocabulary.most_common(n):
            ## otherwise
            if w not in self.stop_words and w not in self.additional_stop_words:
                    w = stemmer.stem(w)
                    processed_text = processed_text + ' ' + w
        return processed_text
