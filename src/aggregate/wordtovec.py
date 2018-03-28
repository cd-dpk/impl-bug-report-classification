import csv
from src.aggregate.pre_processor import TextPreprocessor

class Word2VecRep:

    def __init__(self, data_path):
        self.src_sentences = []
        self.bug_sentences = []
        self.data_path = data_path

    def retrive_sentences_src(self, src_file:str):
        with open('../bug_localization/' + src_file + '_proc.csv', newline='', encoding='UTF-8') as csvfile:
            self.src_sentences= []
            reader = csv.DictReader(csvfile)
            counter = 0
            # ambari_prob = [1329], derby[1698]
            prob = [1696]
            for row in reader:
                if counter in prob:
                    continue
                print(str(counter)+','+(row['class_id'] in (None, '') and '' or row['class_id']))
                line_sentence = []
                textProcessor = TextPreprocessor()
                text = (row['class_content'] in (None, '') and '' or row['class_content'])
                # print(text)
                line_sentence = textProcessor.getProcessedText(text)
                #print(line_sentence)
                self.src_sentences.append(line_sentence)
                counter += 1
        return

    def retrive_sentences_bug(self, bug_file: str):
        with open(self.data_path + bug_file + '_txt_proc.csv', newline='', encoding="UTF-8") as csvfile:
            self.bug_sentences = []
            reader = csv.DictReader(csvfile)
            for row in reader:
                line_sentence = []
                textProcessor = TextPreprocessor()
                #text = (row['summary_col'] in (None, '') and '' or row['summary_col']) + ' ' + (row['description_col'] in (None, '') and '' or row['description_col'])
                text = (row['summary_col'] in (None, '') and '' or row['summary_col'])
                line_sentence = textProcessor.getProcessedText(text)
                self.bug_sentences.append(line_sentence)
        return

    def train_word2vec(self, file:str, src: bool, bug: bool, dim:int=200):
        print(file)
        if bug:
            self.retrive_sentences_bug(file)
            #print("Bug", len(self.bug_sentences))
            # for sentence in self.bug_sentences:
            #     print(sentence)
        if src:
            self.retrive_sentences_src(file)
            #print("Src", len(self.src_sentences))
            # for sentence in self.src_sentences:
            #     print(sentence)

        from gensim.models.word2vec import Word2Vec
        sentences = self.bug_sentences + self.src_sentences
        model = Word2Vec(sentences, size=dim, window=5, min_count=2, workers=4)
        model.wv.save_word2vec_format(self.data_path + file+'_'+str(dim)+'_'+str(src)+'_wv.txt', binary=False)
        print(file+' completed!')
        return True

subjects = ['Camel_Shaon','ambari','derby','wicket']
#subjects = ['wicket']
dims = [200,100,150]
data_path = "F:/DIPOK/simulated_data/"
for subject in subjects:
    for dim in dims:
        import sys
        sys.stdout = open(data_path + subject + '_wv_log.txt', 'w')
        wv = Word2VecRep(data_path=data_path)
        print(subject)
        wv.train_word2vec(subject, False, True, dim=dim)
        sys.stdout.close()

