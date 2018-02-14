import sys,csv,re,stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist


class TextPreprocessor:
    stop_words = set(stopwords.words('english'))
    additional_stop_words = []
    for x in range(26):
        additional_stop_words.append(chr(ord('a') + x))

    def getProcessedText(self, text):
        stemmer = PorterStemmer()
        space_deleted_tokens = re.split('\s',text)
        processed_text = []
        for space_token in space_deleted_tokens:
            # if it is a link remove it
            if re.fullmatch('(https?|ftp|file)://.*', space_token):
                continue
            # other wise continue
            # space_token = re.sub('[_]', ' ', space_token)
            # print(space_token)
            ## camel case and Pascal Case splitted
            case_token = re.sub('(?<=[A-Z])(?=[A-Z][a-z])|(?<=[^A-Z])(?=[A-Z])|(?<=[A-Za-z])(?=[^A-Za-z_])', ' ', space_token)
            if case_token != space_token:
                space_token += ' ' + case_token
            # print(space_token)
            tokens = regexp_tokenize(space_token, pattern='[a-zA-Z_]+')
            # print('Tokens:',tokens)
            for w in tokens:
                if re.fullmatch(".*_$",w):
                     w = re.sub("_","",w)
                if stringcase.lowercase(w) not in self.stop_words and stringcase.lowercase(w) not  in self.additional_stop_words:
                    if re.fullmatch("([A-Za-z]([a-z]+))?", w) or re.fullmatch("[a-z]+", w):
                        processed_text.append(stemmer.stem(w))
                    else:
                        processed_text.append(w)
        return processed_text

