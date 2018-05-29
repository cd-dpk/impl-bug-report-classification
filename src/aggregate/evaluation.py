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
data_path = "/media/geet/Random/PYTHON/simulated_data/"
sampling = [0]
subjects = [ambari, camel_shaon, derby]
false_sec_subjects = [[ambari, 0.05, 0.25], [camel_shaon, 0.75, 0.15], [derby, 0.4, 0.15]]
false_perf_subjects = [[ambari, 0.6, 0.85], [camel_shaon, 0.4, 0.5], [derby, 0.6, 0.5]]
true_sec_subjects = [[ambari, 0.8, 0.25], [camel_shaon, 0.8, 0.25], [derby, 0.35, 0.5]]
true_perf_subjects = [[ambari, 0.35, 0.5], [camel_shaon, 0.8, 0.5], [derby, 0.35, 0.85]]

# for subject in false_perf_subjects:
#     NormalExperiment(data_path=data_path, file=subject[0], intent=Performance).\
#         do_experiment_txt_sampling_feature_selection_final\
#         (des=False, sampling_index=0, hypo=MultinomialNB(), alpha=subject[1], l=subject[2])
#
# for subject in true_perf_subjects:
#     NormalExperiment(data_path=data_path, file=subject[0], intent=Performance).\
#         do_experiment_txt_sampling_feature_selection_final\
#         (des=True, sampling_index=0, hypo=MultinomialNB(), alpha=subject[1], l=subject[2])

# for subject in false_perf_subjects:
#     NormalExperiment(data_path=data_path, file=subject[0], intent=Security).\
#         do_experiment_txt_sampling_classifier(des=False)
#     NormalExperiment(data_path=data_path, file=subject[0], intent=Security). \
#         do_experiment_txt_sampling_classifier(des=True)
#
#     NormalExperiment(data_path=data_path, file=subject[0], intent=Performance).\
#         do_experiment_txt_sampling_classifier(des=False)
#     NormalExperiment(data_path=data_path, file=subject[0], intent=Performance).\
#         do_experiment_txt_sampling_classifier(des=True)

# NormalExperiment(data_path=data_path, file=ambari, intent=Performance).\
#     do_experiment_txt_sampling_feature_selection_final\
#         (des=False, sampling_index=0, hypo=MultinomialNB(), alpha=0.9, l=0.15)
