from src.aggregate.NormalExperiment import NormalExperiment
from src.aggregate.pre_process import Preprocessor
from src.aggregate.vec_rep import VectorRepresenter
from src.aggregate.grep import GREP
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
alphas = [0.4, 0.5, 0.6]
sampling = [0]
subjects = [camel_shaon, derby, ambari, wicket]
for subject in subjects:
    Preprocessor(subject).pre_process(True,True)
    VectorRepresenter(subject).vec_process(False, True, True)
