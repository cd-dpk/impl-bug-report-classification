import sys,csv,re,stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from src.aggregate.pre_processor import TextPreprocessor


class Preprocessor:

    def __init__(self, file):
        self.file = file

        self.perf_keywords = ["performance", "slow", "speed", "latency", "throughput",
                         "cpu", "disk", "memory", "usage", "resource", "calling",
                         "times", "infinite", "loop"]
        self.sec_keywords = ["add", "denial", "service", "XXE", "remote", "open", "redirect", "OSVDB", "vuln", "CVE", "XSS",
                        "ReDoS",
                        "NVD", "malicious", "frame", "attack", "exploit", "directory", "traversal", "RCE", "dos",
                        "XSRF",
                        "clickjack", "session", "fixation", "hijack", "advisory", "insecure", "security", "cross",
                        "origin", "unauthori[z|s]ed", "authenticat(e|ion)", "brute force", "bypass", "credential",
                        "DoS", "expos(e|ing)", "hack", "harden", "injection", "lockout", "over flow", "password", "PoC",
                        "proof", "poison", "privelage", "(in)?secur(e|ity)", "(de)?serializ", "spoof", "traversal"]

        # which single character is removed
        self.additionalStopWords = []
        for x in range(26):
            self.additionalStopWords.append(chr(ord('a') + x))
        self.vocabulary = FreqDist()

    def predict_keywords(self, t):
        # t = re.sub('[_]',' ', t)
        # t = re.sub('(?<=[A-Z])(?=[A-Z][a-z])|(?<=[^A-Z])(?=[A-Z])|(?<=[A-Za-z])(?=[^A-Za-z])',' ',t)
        tokens = regexp_tokenize(t, pattern='[a-zA-Z]+')
        sec_cl = 0
        perf_cl = 0
        for w in tokens:
            # print(w)
            for s in self.sec_keywords:
                s = "(?i)" + s
                if re.search(s, w):
                    sec_cl = 1
                    break
            # print(sec_cl)
            for p in self.perf_keywords:
                p = "(?i)" + p
                if re.search(p, w):
                    perf_cl = 1
                    break
            # print(perf_cl)
            if sec_cl == 1 and perf_cl == 1:
                break
        return (sec_cl, perf_cl)

    def proc_csv_file(self):
        with open('../data/' + self.file + '.csv', newline='', encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile)
            print('issue_id,reporter,component,keywords,summary,description,ST,Patch,CE,TC,EN,files,target_Security,target_Performance')
            for row in reader:
                issue_id = str(row['issue_id'] in (None, '') and '' or row['issue_id'])
                reporter = row['reporter'] in (None, '') and 'null' or row['reporter']
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
                sec, perf = self.predict_keywords(
                    (row['summary'] in (None, '') and '' or '') + " " + (row['description'] in (None, '') and '' or ''))

                print(issue_id + ',' + reporter + ',' + component + ',' + str(sec) + ',' +
                          summary+ ',' + description + ',' + st + ',' + patch + ',' + ce + ','
                          + tc + ',' + en + ',' + files + ',' + security_label + "," + perf_label)

        return

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

    def pre_process(self):
        sys.stdout = open(self.file+'_proc.csv','w',encoding="UTF-8")
        self.proc_csv_file()
        # self.proc_xml_file()
        sys.stdout.close()
        return


