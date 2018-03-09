import os, csv
from src.aggregate.pre_processor import TextPreprocessor
import xml.etree.ElementTree as ET
import re

class Word2VecRep:
    src_sentences = []
    bug_sentences = []

    def __init__(self):
        self.src_sentences = []
        self.bug_sentences = []
    # make word_2_vec from data
    def model_word2vec_DATA(self):
        dirs = ['/media/geet/Random/DATA/XML_LINUX/', '/media/geet/Random/DATA/XML_ECLIPSE/']
        sentences = []
        counter = 0
        for dir in dirs:
            files = os.listdir(dir)
            for file in files:
                print(dir + file)
                tree = ET.parse(dir + file)
                root = tree.getroot()
                for bug in root.findall('bug'):
                    sentence = bug.find('short_desc').text
                    t = TextPreprocessor()
                    line_sentence = []
                    for word in re.split(" ", t.getProcessedText(text=sentence)):
                        line_sentence.append(word)
                    sentences.append(line_sentence)
                    counter += 1

        for sentence in sentences:
            print(sentence)

        from gensim.models.word2vec import Word2Vec
        model = Word2Vec(sentences, size=100, window=5, min_count=2, workers=4)
        model.wv.save_word2vec_format('f.txt', binary=False)

        print("COMPLETE")

    def retrive_sentences_src(self, src_file:str):
        with open('../bug_localization/' + src_file + '_proc.csv', newline='') as csvfile:
            self.src_sentences= []
            reader = csv.DictReader(csvfile)
            counter = 0
            # ambari_prob = [1329], derby[1698]
            prob = [1698]
            for row in reader:
                if counter in prob:
                    continue
                print(counter)
                line_sentence = []
                textProcessor = TextPreprocessor()
                text = (row['class_content'] in (None, '') and '' or row['class_content'])
                # print(text)
                line_sentence = textProcessor.getProcessedText(text)
                print(line_sentence)
                self.src_sentences.append(line_sentence)
                counter += 1
        return

    def retrive_sentences_bug(self, bug_file: str):
        with open(bug_file + '_proc.csv', newline='', encoding="UTF-8") as csvfile:
            self.bug_sentences = []
            reader = csv.DictReader(csvfile)
            for row in reader:
                line_sentence = []
                textProcessor = TextPreprocessor()
                text = (row['proc_summary'] in (None, '') and '' or row['proc_summary'])
                line_sentence = textProcessor.getProcessedText(text)
                self.bug_sentences.append(line_sentence)
        return

    def train_word2vec(self, file:str, src: bool, bug: bool):
        print(file)
        if bug:
            self.retrive_sentences_bug(file)
            print("Bug", len(self.bug_sentences))
            for sentence in self.bug_sentences:
                print(sentence)
        if src:
            self.retrive_sentences_src(file)
            print("Src", len(self.src_sentences))
            # for sentence in self.src_sentences:
            #     print(sentence)

        from gensim.models.word2vec import Word2Vec
        sentences = self.bug_sentences + self.src_sentences
        k =200
        model = Word2Vec(sentences, size=k, window=5, min_count=2, workers=4)
        model.wv.save_word2vec_format(file+'_'+str(k)+'_wv.txt', binary=False)
        return True

subject ='apache'
# import sys
# sys.stdout = open(subject+'log.txt', 'w')
wv = Word2VecRep()
print(subject)
wv.train_word2vec(subject, False, True)
# sys.stdout.close()
