import re
from src.aggregate.pre_processor import TextPreprocessor

var = '2WW_W'

a = re.fullmatch('(([A-Za-z]([a-z]+))+)|([A-Z]+)|([A-Za-z_0-9]+)', var)

if a:
    print('Y', a)
else:
    print('N', a)

link = "https://"

a = re.fullmatch('(https?|ftp|file)://.*',link)

if a:
    print('Y', a)
else:
    print('N', a)


t_p = TextPreprocessor()
var = "HelloWorld getMessage http:// 5 XXX cla_ss {}"
print(t_p.getProcessedText(var))



