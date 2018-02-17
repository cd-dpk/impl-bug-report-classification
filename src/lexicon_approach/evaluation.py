from src.lexicon_approach.pre_process import Preprocessor
from src.lexicon_approach.NormalExperiment import NormalExperiment

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

apache = 'apache'
Security = 'Security'
Performance = 'Performance'

mnb = MultinomialNB()
lr = LogisticRegression()
ada = AdaBoostClassifier()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
supvecmac = svm.SVC(probability=True)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')


# Preprocessor(apache,Security).pre_process()
# VectorRepresenter(apache).vec_process()

# NormalExperiment(apache,Security).do_experiment_txt_feature_selection(l=2000, l1_ratio=0.5, hypo=mnb)
# NormalExperiment(apache,Security).do_experiment_txt_sampling_classifier(hypo=mnb)
# NormalExperiment(apache,Security).do_experiment_feature_terms()
# NormalExperiment(apache,Performance).do_experiment_feature_terms()

# NormalExperiment(apache,Security).do_experiment_featured_terms(mnb)
# NormalExperiment(apache, Security).do_experiment_lexicon()

# Preprocessor('wicket').pre_process()
NormalExperiment('ambari', 'Security').do_experiment_lexicon()