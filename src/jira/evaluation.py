from src.jira.NormalExperiment import NormalExperiment
# from src.lexicon_approach.pre_process import Preprocessor
# from src.jira.vec_rep import VectorRepresenter

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

hypos = [mnb, knn, supvecmac, rf, ada]
# NormalExperiment(apache,Security).do_experiment_txt_sampling_ensemble_voting(sampling_index=2, hypos=hypos)
# NormalExperiment(apache,Security).do_experiment_txt_sampling_ensemble_stacking(sampling_index=2,Hypo=lr, hypos=[mnb,knn,supvecmac,rf,ada])
# NormalExperiment(apache, Security).do_experiment_txt_feature_selection(l=200, selection_method=0, l1_ratio=0.5, hypos=[mnb])
# NormalExperiment(apache, Performance).do_experiment_txt(hypo=mnb)
#NormalExperiment(apache, Performance).do_experiment_txt_feature_selection(mnb, alpha=0.5)
#print("Performance Completed")
'''
alphas = [0.55, 0.45, 0.6]
for x in range(len(alphas)):
    NormalExperiment(apache, Security).do_experiment_txt_feature_selection(mnb, alpha=alphas[x])
    print(alphas[x])
    NormalExperiment(apache, Performance).do_experiment_txt_feature_selection(mnb, alpha=alphas[x])
    print(alphas[x])
'''
alphas = [0.4,0.5,0.6]
sampling = [0]
'''
for x in range(len(alphas)):
    NormalExperiment(apache, Security).do_experiment_txt_feature_selection( mnb, alpha=alphas[x])
    print('Security', 'ratio', alphas[x])
    NormalExperiment(apache, Performance).do_experiment_txt_feature_selection( mnb, alpha=alphas[x])
    print('Performance', 'ratio', alphas[x])
'''
# NormalExperiment(apache, Security).do_experiment_txt_sampling(sampling_index=0, hypo=mnb)
# NormalExperiment(apache, Performance).do_experiment_txt_sampling(sampling_index=0, hypo=mnb)
NormalExperiment(apache, Security).do_experiment_txt(mnb)
NormalExperiment(apache, Performance).do_experiment_txt(mnb)

# NormalExperiment(apache, Security).do_experiment_txt_sampling_classifier(2, mnb)
# NormalExperiment(apache, Security).do_experiment_feature_terms()
# NormalExperiment(apache, Performance).do_experiment_feature_terms()
# NormalExperiment(apache, Security).do_experiment_featured_terms(mnb)
# NormalExperiment(apache, Security).do_experiment_lexicon()
# NormalExperiment(apache, Security).do_experiment_txt_sampling_ensemble_probability(0, hypos=[mnb])
# Preprocessor('ambari').pre_process

# NormalExperiment(apache, Security).do_experiment_txt_after_feature_selected(1000,0.5,supvecmac)
# NormalExperiment(apache,Security).do_experiment_txt(mnb)
# NormalExperiment(apache, Performance).do_experiment_generate_lexicon_terms()