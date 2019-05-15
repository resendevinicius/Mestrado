from skmultilearn.dataset import load_dataset_dump
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.adapt import MLkNN
from skmultilearn.dataset import load_dataset
from sklearn.svm import SVC
import random
import sklearn.metrics as metrics
from sklearn import tree
random.seed(1)

dataset = raw_input("Nome da base: ")
print dataset
train = load_dataset_dump(dataset + '-train.scikitml.bz2')
test = load_dataset_dump(dataset + '-test.scikitml.bz2')
#best = 0
#id = 0.0
#for i in range (1, 20) :
#	classifier = MLkNN(k = i)
#	prediction = classifier.fit(train['X'], train['y']).predict(test['X'])
#	if (1 - metrics.hamming_loss(prediction, test['y']) > best) :
#		best = 1 - metrics.hamming_loss(prediction, test['y'])
#		id = i;
#
#classifier = MLkNN(k = id)
#prediction = classifier.fit(train['X'], train['y']).predict(test['X'])
#print classifier
#print 'Subset Accuracy: ', metrics.accuracy_score(prediction, test['y'])
#print 'Hamming Loss: ', metrics.hamming_loss(prediction, test['y'])
#print 'Accuracy: ', 1 - metrics.hamming_loss(prediction, test['y'])

classifier = ClassifierChain(SVC())
prediction = classifier.fit(train['X'], train['y']).predict(test['X'])
print'------------------------------------------'
print classifier
print 'Subset Accuracy: ', metrics.accuracy_score(prediction, test['y'])
print 'Hamming Loss: ', metrics.hamming_loss(prediction, test['y'])
print 'Accuracy: ', 1 - metrics.hamming_loss(prediction, test['y'])

classifier = BinaryRelevance(SVC())
prediction = classifier.fit(train['X'], train['y']).predict(test['X'])
print'------------------------------------------'
print classifier
print 'Subset Accuracy: ', metrics.accuracy_score(prediction, test['y'])
print 'Hamming Loss: ', metrics.hamming_loss(prediction, test['y'])
print 'Accuracy: ', 1 - metrics.hamming_loss(prediction, test['y'])

