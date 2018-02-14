from src.aggregate.NormalExperiment import NormalExperiment
from src.aggregate.pre_process import Preprocessor
from src.aggregate.vec_rep import VectorRepresenter

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

NormalExperiment(camel, Security).do_experiment_txt_fs(supvecmac)

# Preprocessor(wicket, Security).pre_process()
# VectorRepresenter(ambari, Security).vec_process()
# VectorRepresenter(camel, Security).vec_process()
# VectorRepresenter(derby, Security).vec_process()
# VectorRepresenter(derby, Security).vec_process()
# VectorRepresenter(wicket, Security).vec_process()


# NormalExperiment(ambari, Security).do_experiment_txt(supvecmac)
# NormalExperiment(ambari,Security).do_experiment_txt(supvecmac)