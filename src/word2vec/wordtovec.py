import os
from src.aggregate.pre_processor import TextPreprocessor
import xml.etree.ElementTree as ET
import re
dirs = ['/media/geet/Random/DATA/XML_LINUX/','/media/geet/Random/DATA/XML_ECLIPSE/']
sentences = []
counter = 0
for dir in dirs:
    files = os.listdir(dir)
    for file in files:
        print(dir+file)
        tree = ET.parse(dir+file)
        root = tree.getroot()
        for bug in root.findall('bug'):
            sentence = bug.find('short_desc').text
            t = TextPreprocessor()
            line_sentence = []
            for word in re.split(" ",t.getProcessedText(text=sentence)):
                line_sentence.append(word)
            sentences.append(line_sentence)
            counter += 1
    break
for sentence in sentences:
    print(sentence)

from gensim.models.word2vec import Word2Vec
model = Word2Vec(sentences, size=100, window=5, min_count=2, workers=4)
model.wv.save_word2vec_format('f.txt', binary=False)

a = model.wv.word_vec("host")
print(len(a))

