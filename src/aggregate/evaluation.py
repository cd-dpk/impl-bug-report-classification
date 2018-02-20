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
camel_shaon = 'Camel_Shaon'

mnb = MultinomialNB()
lr = LogisticRegression()
ada = AdaBoostClassifier()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
supvecmac = svm.SVC(probability=True)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Preprocessor(camel_shaon, Security).pre_process()
# Preprocessor(wicket).pre_process()
# Preprocessor(derby, Security).pre_process()
# Preprocessor(wicket, Security).pre_process()

# VectorRepresenter(camel_shaon, Security).vec_process()
# VectorRepresenter(wicket).vec_process()
# VectorRepresenter(derby, Security).vec_process()
# VectorRepresenter(wicket, Security).vec_process()
# VectorRepresenter(wicket, Security).vec_process()


# NormalExperiment(ambari, Security).do_experiment_txt_sampling_classifier(sampling_index=1,hypo=supvecmac)
# NormalExperiment(ambari,Security).do_experiment_txt_feature_selection(1000,0.5,mnb)
NormalExperiment(camel,Security).do_experiment_tex_src(0, mnb)
# from src.bug_localization.BugLocatorlExperiment import BugLocatorExperiment
# x = BugLocatorExperiment('camel', 'security')