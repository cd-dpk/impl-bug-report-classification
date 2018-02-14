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


# Preprocessor(derby, Surprising).pre_process()
# VectorRepresenter(derby, Surprising).vec_process()
# VectorRepresenter(camel, Security).vec_process()
# VectorRepresenter(derby, Security).vec_process()
# VectorRepresenter(derby, Security).vec_process()
# VectorRepresenter(wicket, Security).vec_process()


NormalExperiment(camel, Surprising).do_experiment_txt(mnb)
# NormalExperiment(ambari,Security).do_experiment_txt(supvecmac)