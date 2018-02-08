from src.aggregate.experiment import Experiment
from src.aggregate.NormalExperiment import NormalExperiment
from src.aggregate.VotingExperiment import VotingExperiment
from src.aggregate.StackingExperiment import StackingExperiment

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
mnb = MultinomialNB()
lr = LogisticRegression()
ada = AdaBoostClassifier()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
supvecmac = svm.SVC()
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

exp = NormalExperiment('ambari')
# exp.doExperiment(hypo=mnb)
exp.do_experiment_structural(h_f=supvecmac,h_s=dt)



