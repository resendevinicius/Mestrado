from sklearn.svm import SVC
from skmultilearn.dataset import load_dataset_dump
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn import tree
from skmultilearn.adapt import MLkNN
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import ComplexMul as cm
import numpy as np
import sklearn.metrics as metrics
import operator

X_train, y_train, features_train, labels_train = load_dataset_dump('scikit_ml_learn_data/emotions-train.scikitml.bz2')
X_test, y_test, features_test, labels_test = load_dataset_dump('scikit_ml_learn_data/emotions-test.scikitml.bz2')

X = np.concatenate((X_train.toarray(), X_test.toarray()))
y = np.concatenate((y_train.toarray(), y_test.toarray()))

parameters = {'k' : [5, 7, 10, 13], 'lambd' : [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 'threshold': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'classifier': [MLkNN()]}
score = 'accuracy'
ks = parameters['k']
lambds = parameters['lambd']
thresholds = parameters['threshold']
classifiers = parameters['classifier']
number_splits = 5
kf = KFold(n_splits = number_splits)


scores = {}

for k in ks:
  for lambd in lambds:
    for threshold in thresholds:
      parameters_ = 'k: ' + str(k) + ' Lambda: ' + str(lambd) + ' Threshold: ' + str(threshold) + ' Classifier: MLkNN'
      scores.update({parameters_: 0})

for train_index, test_index in kf.split(X):
  for k in ks:
    for lambd in lambds:
      for threshold in thresholds:
        model = cm.ComplexMul(k = k, classifier = classifiers[0], lambd = lambd, threshold = threshold)
        model.fit(X[train_index], y[train_index])
        parameters_ = 'k: ' + str(k) + ' Lambda: ' + str(lambd) + ' Threshold: ' + str(threshold) + ' Classifier: MLkNN'
        value = (metrics.accuracy_score(model.predict(X[test_index]).flatten(), y[test_index].flatten()))
        scores[parameters_] += value
      
ans = max(scores.items(), key = operator.itemgetter(1))

print('Avg Score: ' + str(ans[1] / number_splits) + '\n' + str(ans[0]))

