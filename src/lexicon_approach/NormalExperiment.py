from builtins import list
import math
from sklearn.naive_bayes import MultinomialNB
from src.jira.experiment import Experiment
from src.aggregate.pre_processor import TextPreprocessor
from collections import Counter
import numpy as np

class NormalExperiment(Experiment):

    def do_experiment_lexicon(self):
        # read all the lexicons
        import csv
        positive_lexicon = []
        negative_lexicon = []
        neutral_lexicon = []

        csvfile = open('../jira/apache_'+self.intent+'_pos_terms.txt',newline='')
        reader = csv.DictReader(csvfile)
        for row in reader:
            if float(row['score']) > 0.0:
                positive_lexicon.append([row['index'], row['term'], row['score']])
        csvfile.close()

        csvfile = open('../jira/apache_' + self.intent + '_neg_terms.txt', newline='')
        reader = csv.DictReader(csvfile)
        counter = 0
        for row in reader:
            # if counter >= 1000:
            #     break
            if float(row['score']) > 0.0:
                negative_lexicon.append([row['index'], row['term'], row['score']])
                counter += 1
        csvfile.close()

        csvfile = open(
            '../jira/apache_' + self.intent + '_neu_terms.txt',
            newline='')
        reader = csv.DictReader(csvfile)
        counter = 0
        for row in reader:
            # if counter >= 1000:
            #     break
            if float(row['score']) == 0.0:
                neutral_lexicon.append([row['index'], row['term'], row['score']])
                counter += 1
        csvfile.close()

        print((positive_lexicon))
        print((negative_lexicon))
        print((neutral_lexicon))


        # exit(400)
        # all processed file

        y_test = []
        y_predict = []

        with open(self.file + '_lexicon_proc.csv', newline='', encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile)
            counter = 0
            for row in reader:
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

                label = 0
                if self.intent == 'Security':
                    label = str((row['target_Security'] in (None, '') and '0' or row['target_Security']))
                elif self.intent == 'Performance':
                    label = str((row['target_Performance'] in (None, '') and '0' or row['target_Performance']))

                test = int(label)
                y_test.append(test)

                # using lexicons
                w_one, w_zero = (0.0, 0.0)
                # w_one -> sum of positive lexicons
                # w_zero -> sum of negative lexicons

                # terms = TextPreprocessor().term_count(summary+" "+description)
                terms = TextPreprocessor().term_count(summary)
                for term in terms:
                    for lexicon in positive_lexicon:
                        if term[0] == lexicon[1]:
                            w_one += float(term[1]) * float(lexicon[2])
                            # w_one += float(term[1])
                            # w_one += 1
                            break

                for term in terms:
                    for lexicon in negative_lexicon:
                        if term[0] == lexicon[1]:
                            w_zero += float(term[1]) * float(lexicon[2])
                            # w_zero += float(term[1])
                            # w_zero += 1
                            break


                # predict using lexicon decison rule w_one > w_zero

                # w_one = math.pow(math.e, w_one)
                predict = 0
                if w_one > w_zero:
                    predict = 1
                    y_predict.append(predict)
                else:
                    predict = 0
                    y_predict.append(predict)

                counter += 1
                print(counter, w_one, w_zero, test, predict)


        print(len(y_test))
        t_p, t_n, f_p, f_n = self.calc_tuple(self.confusion_matrix(y_test, y_predict))
        print(t_p, t_n, f_p, f_n)
        print(self.calc_acc_pre_rec({'t_p': t_p, 'f_p': f_p, 't_n': t_n, 'f_n': f_n}))
        return