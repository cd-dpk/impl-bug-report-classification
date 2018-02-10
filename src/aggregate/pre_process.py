import sys,csv,re,stringcase
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from src.aggregate.pre_processor import TextPreprocessor


class Preprocessor:

    def __init__(self, file, intent):
        self.file = file
        self.intent = intent
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

    def proc_file(self):
        with open('../data/' + self.file + '.csv', newline='', encoding="UTF-8") as csvfile:
            reader = csv.DictReader(csvfile)
            print('issue_id,reporter,component,keywords,summary,description,ST,Patch,CE,TC,EN,files,'+self.intent)
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
                label = str((row[self.intent] in (None, '') and '0' or row[self.intent]))
                st = str((row['ST'] in (None, '') and '0' or row['ST']))
                patch = str((row['Patch'] in (None, '') and '0' or row['Patch']))
                ce = str((row['CE'] in (None, '') and '0' or row['CE']))
                tc = str((row['TC'] in (None, '') and '0' or row['TC']))
                en = str((row['EN'] in (None, '') and '0' or row['EN']))
                files = row['files'] in (None, '') and '' or row['files']
                sec, perf = self.predict_keywords(
                    (row['summary'] in (None, '') and '' or '') + " " + (row['description'] in (None, '') and '' or ''))
                if self.intent == 'Security':
                    print(issue_id + ',' + reporter + ',' + component + ',' + str(sec) + ',' +
                          summary+ ',' + description + ',' + st + ',' + patch + ',' + ce + ','
                          + tc + ',' + en + ',' + files + ',' + label)
                elif self.intent == 'Performance':
                    print(issue_id + ',' + reporter + ',' + component + ',' + str(perf) + ',' +
                          summary + ',' + description + ',' + st + ',' + patch + ',' + ce + ',' +
                          tc + ',' + en + ',' + files + ',' + label)
        return

    def pre_process(self):
        sys.stdout= open(self.file+'_'+self.intent+'_proc.csv','w',encoding="UTF-8")
        self.proc_file(self.file)
        sys.stdout.close()
        return

