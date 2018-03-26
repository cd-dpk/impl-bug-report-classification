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
# '''
alphas = [0.4, 0.5, 0.6]
sampling = [0]
subjects = [camel_shaon, derby, ambari, wicket]

# VectorRepresenter(derby).vec_process()
# NormalExperiment(ambari, Security).do_experiment_first_txt_second_categorical_weka(hypo1=mnb)
# NormalExperiment(ambari, Performance).do_experiment_first_txt_second_categorical_weka(hypo1=mnb)

# NormalExperiment(camel_shaon, Security).do_experiment_first_txt_second_categorical_weka(hypo1=mnb)
# NormalExperiment(camel_shaon, Performance).do_experiment_first_txt_second_categorical_weka(hypo1=mnb)

# NormalExperiment(derby, Security).do_experiment_first_txt_second_categorical_weka(hypo1=mnb)
# NormalExperiment(derby, Performance).do_experiment_first_txt_second_categorical_weka(hypo1=mnb)

# NormalExperiment(wicket, Security).do_experiment_first_txt_second_categorical_weka(hypo1=mnb)
# NormalExperiment(wicket, Performance).do_experiment_first_txt_second_categorical_weka(hypo1=mnb)

# NormalExperiment(camel_shaon, Security).do_experiment_txt_sampling_classifier(sampling_index=0, hypo=mnb)
