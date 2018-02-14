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
        with open('/media/geet/Files/IITDU/MSSE-03/SRC_P/' + src_file + '_term.csv', newline='') as csvfile:
            self.src_sentences= []
            reader = csv.DictReader(csvfile)
            counter = 0
            ambari_prob = [1329]
            for row in reader:
                # print(counter)
                line_sentence = []
                textProcessor = TextPreprocessor()
                text = (row['proc'] in (None, '') and '' or row['proc'])
                # print(text)
                line_sentence = textProcessor.getProcessedText(text)
                self.src_sentences.append(line_sentence)
                counter += 1
        return

    def retrive_sentences_bug(self,bug_file: str):
        with open('../data/' + bug_file + '.csv', newline='') as csvfile:
            self.bug_sentences= []
            reader = csv.DictReader(csvfile)
            for row in reader:
                line_sentence = []
                textProcessor = TextPreprocessor()
                text = (row['summary'] in (None, '') and '' or row['summary']) + ' ' + (row['description'] in (None, '') and '' or row['description'])
                line_sentence = textProcessor.getProcessedText(text)
                self.bug_sentences.append(line_sentence)
        return

    def train_word2vec(self, file:str, src: bool, bug: bool):
        print(file)
        if bug:
            self.retrive_sentences_bug(file)
            print("Bug",len(self.bug_sentences))
            # for sentence in self.bug_sentences:
            #     print(sentence)
        if src:
            self.retrive_sentences_src(file)
            print("Src", len(self.src_sentences))
            # for sentence in self.src_sentences:
            #     print(sentence)

        from gensim.models.word2vec import Word2Vec
        sentences = self.bug_sentences + self.src_sentences
        model = Word2Vec(sentences, size=100, window=5, min_count=2, workers=4)
        model.wv.save_word2vec_format(file+'_wv.txt', binary=False)
        return True

import sys
# sys.stdout = open('camel'+'log.txt', 'w')
wv = Word2VecRep()
wv.train_word2vec('camel', True, True)
# sys.stdout.close()
