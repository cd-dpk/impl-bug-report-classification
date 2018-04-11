import sys, csv, re, stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from src.aggregate.grep import GREP
from src.aggregate.pre_processor import TextPreprocessor

class Preprocessor:

    def __init__(self, data_path, file):
        self.file = file
        self.data_path = data_path

    def get_team(self):
        team = []
        with open(self.data_path + self.file + '_team.csv', newline='', encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                team.append(row['team'])
        return team

    # load the lexicon data
    def load_lexicon_data(self, intent):
        import csv
        positive_lexicon = {}
        negative_lexicon = {}
        neutral_lexicon = {}

        csvfile = open('../jira/apache_'+intent+'_pos_terms.txt', newline='')
        reader = csv.DictReader(csvfile)
        for row in reader:
            if float(row['score']) >= 0.0:
                positive_lexicon.__setitem__(row['term'], row['score'])
        csvfile.close()

        csvfile = open('../jira/apache_'+intent+'_neg_terms.txt', newline='')
        reader = csv.DictReader(csvfile)
        # counter = 0
        for row in reader:
            # if counter >= 1000:
            #     break
            if float(row['score']) >= 0.0:
                negative_lexicon.__setitem__(row['term'], row['score'])
                # counter += 1
        csvfile.close()

        csvfile = open('../jira/apache_'+intent+'_neu_terms.txt', newline='')
        reader = csv.DictReader(csvfile)

        for row in reader:
            if float(row['score']) == 0.0:
                neutral_lexicon.__setitem__(row['term'], row['score'])

        csvfile.close()
        return (positive_lexicon, neutral_lexicon, negative_lexicon)

    # process csv file
    def proc_csv_file_txt(self):
        txt_file = open(self.data_path + self.file+'_txt_proc.csv', 'w', encoding='UTF-8')
        with open('../data/' + self.file + '.csv', newline='', encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile)
            txt_file.write('issue_id,summary_col,description_col,target_Security,target_Performance'+'\n')
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
                security_label = (row['Security'] in (None, '') and '' or row['Security'])
                perf_label = (row['Performance'] in (None, '') and '' or row['Performance'])
                txt_file.write(issue_id + ',' +summary + ',' + description + ',' + security_label + "," + perf_label+'\n')
        txt_file.close()
        return

    # process csv file
    def proc_csv_file_str(self):
        # sec_pos_lex, sec_neu_lex, sec_neg_lex = self.load_lexicon_data('Security')
        # perf_pos_lex, perf_neu_lex, perf_neg_lex = self.load_lexicon_data('Performance')
        team = self.get_team()
        str_file = open(self.data_path + self.file+'_str_proc.csv', 'w', encoding='UTF-8')

        with open('../data/' + self.file + '.csv', newline='', encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile)
            # str_file.write()
            str_file.write('issue_id,reporter_col,team_col,component_col,grep_sec,grep_perf,ST_col,Patch_col,CE_col,TC_col,EN_col,files_col,target_Security,target_Performance\n')
            for row in reader:
                issue_id = str(row['issue_id'] in (None, '') and '' or row['issue_id'])
                reporter = row['reporter'] in (None, '') and 'null' or row['reporter']
                team_col = '0'
                if reporter in team:
                    team_col = '1'

                component = row['component'] in (None, '') and 'null' or row['component']
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
                st = str((row['ST'] in (None, '') and '0' or row['ST']))
                patch = str((row['Patch'] in (None, '') and '0' or row['Patch']))
                ce = str((row['CE'] in (None, '') and '0' or row['CE']))
                tc = str((row['TC'] in (None, '') and '0' or row['TC']))
                en = str((row['EN'] in (None, '') and '0' or row['EN']))
                files = row['files'] in (None, '') and '' or row['files']
                terms = TextPreprocessor().term_count(summary+" "+description)
                '''
                sec_pos,sec_neu,sec_neg = (0.0, 0.0, 0.0)
                perf_pos, perf_neu, perf_neg = (0.0, 0.0, 0.0)
                for term in terms:
                    flag_1 = False
                    flag_2 = False

                    for lexicon in sec_pos_lex.keys():
                        if term[0] == lexicon:
                            # presence
                            sec_pos += 1
                            # occurnece
                            # sec_pos += term[1]
                            # weight
                            # sec_pos += float(term[1]) * float(sec_pos_lex[lexicon])
                            flag_1 = True
                            break

                    for lexicon in sec_neu_lex.keys():
                        if flag_1 == True:
                            break
                        if term[0] == lexicon:
                            # presence
                            sec_neu += 1
                            # occurnece
                            # sec_neu += term[1]
                            # weight
                            # sec_neu += float(term[1]) * float(sec_neu_lex[lexicon])
                            flag_1 = True
                            break

                    for lexicon in sec_neg_lex.keys():
                        if flag_1 == True:
                            break
                        if term[0] == lexicon:
                            # presence
                            sec_neg += 1
                            # occurnece
                            # sec_neg += term[1]
                            # weight
                            # sec_neg += float(term[1]) * float(sec_neg_lex[lexicon])
                            flag_1 = True
                            break

                    for lexicon in perf_pos_lex.keys():
                        if term[0] == lexicon:
                            # presence
                            perf_pos += 1
                            # occurnece
                            # perf_pos += term[1]
                            # weight
                            # perf_pos += float(term[1]) * float(perf_pos_lex[lexicon])
                            flag_2 = True
                            break

                    for lexicon in perf_neu_lex.keys():
                        if flag_2 == True:
                            break
                        if term[0] == lexicon:
                            # presence
                            perf_neu += 1
                            # occurnece
                            # perf_neu += term[1]
                            # weight
                            # perf_neu += float(term[1]) * float(perf_neu_lex[lexicon])
                            break

                    for lexicon in perf_neg_lex:
                        if flag_2 == True:
                            break
                        if term[0] == lexicon:
                            # presence
                            perf_neg += 1
                            # occurnece
                            # perf_neg += term[1]
                            # weight
                            # perf_neg += float(term[1]) * float(perf_neg_lex[lexicon])
                            flag_2 = True
                            break
                '''
                sec_grep, sec_predict_label = GREP().predict_security_label(summary=summary, description=description)
                perf_grep, perf_predict_label = GREP().predict_performance_label(summary=summary, description=description)

                str_file.write(issue_id + ',' + reporter + ','+team_col + ',' + component + ',' + str(sec_predict_label) + "," + str(perf_predict_label) + ',' + st + ',' + patch + ',' + ce + ','
                          + tc + ',' + en + ',' + files + ',' + security_label + "," + perf_label+'\n')
        str_file.close()
        return

    #  process all the xml files of apache jira
    def proc_xml_file(self):
        import xml.etree.ElementTree as ET
        import os
        sys.stdout = open(self.file + '_' + self.intent + '_proc.csv', 'w', encoding="UTF-8")
        # print('issue_id,summary,description,Security'+self.intent)
        dir = '/media/geet/Files/IITDU/MSSE-03/DocumentSimilarity-master/Apache/'
        # sec_keys = []
        # with open("/media/geet/Files/IITDU/MSSE-03/DocumentSimilarity-master/sec.txt", "r") as myfile:
        #     for line in myfile:
        #         sec_keys.append(re.sub("\n", "", line))
        #
        # print(sec_keys)
        # perf_keys = []
        # with open("/media/geet/Files/IITDU/MSSE-03/DocumentSimilarity-master/perf.txt", "r") as myfile:
        #     for line in myfile:
        #         perf_keys.append(re.sub("\n", "", line))
        # print(perf_keys)
        print('id,key,summary,proc_summary,sec,perf')
        files = os.listdir(dir)
        sentences = []
        seen_ids = []
        for file in files:
            print(file)
            if re.fullmatch(".*com.xml$",file):
                420
            else:
                continue
            print(dir + file)
            tree = ET.parse(dir + file)
            root = tree.getroot()
            for bug in root.findall('item'):
                id = bug.find('id').text
                if id not in seen_ids:
                    key = bug.find('key').text
                    summary = bug.find('summary').text
                    description = bug.find("description").text
                    sec_flag = bug.find("sec").text
                    perf_flag = bug.find("perf").text
                    t_p = TextPreprocessor()

                    proc_summary = ' '
                    for word in t_p.getProcessedText(summary):
                        proc_summary += ' ' + word

                    # proc_description = ' '
                    # for word in t_p.getProcessedText(description):
                    #     proc_description += ' ' + word

                    # print(id,key,summary,proc_summary,description,proc_description,sec_flag,perf_flag)
                    print(id, key, summary, proc_summary, sec_flag, perf_flag)
                    # print(seen_ids)
                    seen_ids.append(id)
                # t = TextPreprocessor()
                # line_sentence = []
                # for word in re.split(" ", t.getProcessedText(text=sentence)):
                #     line_sentence.append(word)
                # sentences.append(line_sentence)

        sys.stdout.close()
        return

    def pre_process(self, str:bool, txt:bool):
        if str:
            self.proc_csv_file_str()
        if txt:
            self.proc_csv_file_txt()
        return



