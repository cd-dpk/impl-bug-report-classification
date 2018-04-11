import csv

class Converter:

    def __init__(self):
            2
    def convert_csv_to_arff(self, csv_file_name, arff_file_name):
        data_path = "/media/geet/Random/PYTHON/simulated_data/"
        csv_file = open(data_path+csv_file_name+'.csv',newline='',encoding="UTF-8")
        arff_file = open(data_path+arff_file_name+'.arff', 'w')
        arff_file.write('@Relation '+csv_file_name+'\n')
        reader = csv.DictReader(csv_file)
        attributes = reader.fieldnames
        data = '@data\n'
        author = []
        for row in reader:
            author.append((row['auth']))
            data += row['auth']+','+row['team']+','
            data += row['comp0']+','+row['comp1']+','+row['comp2']+',' +\
                    row['comp3']+','+row['comp4']+','+row['comp5']+','
            # '''
            data += row['comp6']+','+row['comp7']+','+row['comp8']+','
            # '''
            '''
            data += row['comp9'] + ',' + row['comp10'] + ',' + row['comp11'] + ',' + row['comp12'] + ',' + \
                    row['comp13'] + ','
            '''
            data += row['grep'] + ',' + row['lex0'] + ','
            data += row['des0'] + ',' + row['des2'] + ',' + row['des3'] + ',' + row['des4'] + ','+\
                    row['prob0']+','+row['prob1']+','+row['target']+'\n'

        author = list(frozenset(author))
        author_data = '@attribute auth {'
        len_author = len(author)
        for x in range(len(author)):
            author_data += author[x]
            if x < len_author - 1:
                author_data +=','
        author_data+='}\n'
        arff_file.write(author_data)
        arff_file.write('@attribute team {0.0,1.0}\n')
        arff_file.write('@attribute comp0 {0.0,1.0}\n')
        arff_file.write('@attribute comp1 {0.0,1.0}\n')
        arff_file.write('@attribute comp2 {0.0,1.0}\n')
        arff_file.write('@attribute comp3 {0.0,1.0}\n')
        arff_file.write('@attribute comp4 {0.0,1.0}\n')
        arff_file.write('@attribute comp5 {0.0,1.0}\n')
        # '''
        arff_file.write('@attribute comp6 {0.0,1.0}\n')
        arff_file.write('@attribute comp7 {0.0,1.0}\n')
        arff_file.write('@attribute comp8 {0.0,1.0}\n')
        # '''
        '''
        arff_file.write('@attribute comp9 {0.0,1.0}\n')
        arff_file.write('@attribute comp10 {0.0,1.0}\n')
        arff_file.write('@attribute comp11 {0.0,1.0}\n')
        arff_file.write('@attribute comp12 {0.0,1.0}\n')
        arff_file.write('@attribute comp13 {0.0,1.0}\n')
        '''
        arff_file.write('@attribute grep {0.0,1.0}\n')
        arff_file.write('@attribute lex0 numeric\n')

        arff_file.write('@attribute des0 {0.0,1.0}\n')
        arff_file.write('@attribute des2 {0.0,1.0}\n')
        arff_file.write('@attribute des3 {0.0,1.0}\n')
        arff_file.write('@attribute des4 {0.0,1.0}\n')
        arff_file.write('@attribute prob0 numeric\n')
        arff_file.write('@attribute prob1 numeric\n')
        arff_file.write('@attribute target {0,1}\n')
        arff_file.write(data)
        arff_file.close()
        return

ambari = 'weka/ambari/'
camel = 'weka/Camel_Shaon/'
derby = 'weka/derby/'
Security = 'Security'
Performance = 'Performance'
intent = Security
subject = camel
des = [False, True]
for x in range(10):
    for y in range(len(des)):
        Converter().convert_csv_to_arff(subject+str(x)+'_'+intent+'_'+str(des[y])+'_train_str', subject+str(x)+'_'+intent+'_'+str(des[y])+'_train_str')
        Converter().convert_csv_to_arff(subject+str(x)+'_'+intent+'_'+str(des[y])+'_test_str', subject+str(x)+'_'+intent+'_'+str(des[y])+'_test_str')
