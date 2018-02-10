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

mnb = MultinomialNB()
lr = LogisticRegression()
ada = AdaBoostClassifier()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
supvecmac = svm.SVC(probability=True)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

pre_ambari = Preprocessor(file='ambari', intent='Security')

exp_ambari = NormalExperiment('camel_shaon_774')
exp_ambari.test()
# exp_camel = NormalExperiment('camel')
# exp_derby = NormalExperiment('derby')
# exp_wicket = NormalExperiment('wicket')
# exp_ambari.do_voting_experiment_txt(hypos=[mnb, ada, rf, knn, supvecmac])
# exp_ambari.do_voting_experiment_str(hypos=[mnb])
# exp_ambari.do_stacking_experiment_txt(Hypo=lr, hypos=[mnb,ada, rf, knn, supvecmac])

# exp_ambari.do_voting_experiment_txt([supvecmac])
# exp_camel.do_voting_experiment_txt([supvecmac])
# exp_derby.do_voting_experiment_txt([supvecmac])
# exp_wicket.do_voting_experiment_txt([knn,mnb,supvecmac])

# exp_ambari.do_stacking_experiment_txt(lr,[knn,mnb,supvecmac])