from src.aggregate.NormalExperiment import NormalExperiment
from src.aggregate.pre_process import Preprocessor
from src.aggregate.vec_rep import VectorRepresenter
from sklearn.naive_bayes import MultinomialNB, GaussianNB
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
camel_shaon = 'Camel_Shaon'
gnb = GaussianNB()
mnb = MultinomialNB()
lr = LogisticRegression()
ada = AdaBoostClassifier()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
supvecmac = svm.SVC(probability=True)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Preprocessor(camel_shaon).pre_process()
# VectorRepresenter(camel).vec_process(True)
# VectorRepresenter(camel_shaon).vec_process(True)

# NormalExperiment(camel_shaon, Security).do_experiment_txt_feature_selection(1500, 0.5, mnb)
# NormalExperiment(camel, Performance).do_experiment_txt_sampling_classifier(0, mnb)
# NormalExperiment(derby, Performance).do_experiment_txt_sampling_classifier(0, mnb)
# NormalExperiment(wicket, Performance).do_experiment_txt_sampling_classifier(0, mnb)
# print(Security)
# NormalExperiment(camel_shaon, Security).do_experiment_txt_sampling_ensemble_stacking(0,lr,[mnb,supvecmac,knn,rf,ada])