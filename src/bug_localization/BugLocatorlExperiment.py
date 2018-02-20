import math, csv

from src.aggregate.pre_processor import TextPreprocessor


class BugLocatorExperiment():

    def __init__(self, file, intent):
        self.file = file
        self.intent = intent
        self.src_terms = {}
        with open("/media/geet/Files/IITDU/MSSE-03/implementation/src/bug_localization/" + self.file + "_idf.csv",
                  newline='') as idf_csvfile:
            idf_reader = csv.DictReader(idf_csvfile)
            for idf_row in idf_reader:
                term = str(idf_row['term'] in (None, '') and '' or idf_row['term'])
                idf = float(idf_row['idf'] in (None, '') and '0' or idf_row['idf'])
                term_obj = {term: idf}
                print(term_obj)
                self.src_terms.__setitem__(term, idf)

    def do_experiment_bug_locate(self, bug_report: dict):
        summary = (bug_report['summary'])
        description = (bug_report['description'])
        label = bug_report['target_Security']
        print(summary, label)

        src_files = []
        with open("/media/geet/Files/IITDU/MSSE-03/implementation/src/bug_localization/" + self.file + "_proc.csv",
                newline='') as src_csvfile:
            src_reader = csv.DictReader(src_csvfile)
            # print('class_id,class_name,class_content')
            for src_row in src_reader:
                class_id = int(src_row['class_id'] in (None, '') and '' or src_row['class_id'])
                clas_content = src_row['class_name'] in (None, '') and ' ' or src_row['class_name']
                class_name = src_row['class_content'] in (None, '') and ' ' or src_row['class_content']
                # print("\t",class_id,class_name, clas_content)

                src_components = TextPreprocessor().term_count(clas_content)
                brs_components = TextPreprocessor().term_count(summary+' '+ description)

                value_src = 0.0
                value_br = 0.0
                prod = 0.0

                for src_component in src_components:
                    if src_component[0] in self.src_terms.keys():
                        value_src += src_component[1] * src_component[1] * self.src_terms[src_component[0]]* self.src_terms[src_component[0]]
                value_src = math.sqrt(value_src)

                for br_component in brs_components:
                    value_br += br_component[1] * br_component[1]
                value_br = math.sqrt(value_br)

                for src_component in src_components:
                    for br_component in brs_components:
                        if src_component[0] == br_component[0]:
                            if src_component[0] in self.src_terms.keys():
                                prod += src_component[1] * br_component[1] * self.src_terms[src_component[0]]

                cosine = prod / (value_src * value_br)
                print(value_src, value_br, prod, cosine)

                src_files.append({'class_id': class_id, 'class_name': class_name, 'cosine': cosine})

        src_files = sorted(src_files, key=lambda sc: sc['cosine'], reverse=True)
        if len(src_files) != 0:
            print(summary, src_files[0])

        return src_files


