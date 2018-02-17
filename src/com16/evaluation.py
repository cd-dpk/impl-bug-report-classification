from src.com16.NormalExperiment import NormalExperiment
from src.com16.pre_process import Preprocessor
from src.com16.vec_rep import VectorRepresenter

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

camel_shaon = "Camel_Shaon"
ambari = 'ambari'
camel = 'camel'
derby = 'derby'
wicket = 'wicket'
Surprising = 'Surprising'
Security = 'Security'
Performance = 'Performance'

mnb = MultinomialNB()
lr = LogisticRegression()
ada = AdaBoostClassifier()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
supvecmac = svm.SVC(probability=True)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')


# Preprocessor(camel_shaon, Security).pre_process()
# VectorRepresenter(camel_shaon, Security).vec_process()

# Preprocessor(camel, Surprising).pre_process()
# VectorRepresenter(camel, Surprising).vec_process()

# Preprocessor(derby, Surprising).pre_process()
# VectorRepresenter(derby, Surprising).vec_process()

# Preprocessor(wicket, Surprising).pre_process()
# VectorRepresenter(wicket, Surprising).vec_process()


NormalExperiment(camel_shaon, Security).do_experiment(supvecmac)
# NormalExperiment(camel,Security).do_experiment_txt(supvecmac)
# NormalExperiment(derby,Security).do_experiment_txt(supvecmac)
# NormalExperiment(wicket,Security).do_experiment_txt(supvecmac)