from collections import defaultdict
import math
import numpy as np
import igraph as ig
from sklearn.metrics import euclidean_distances
from scipy.spatial import distance
from skmultilearn.adapt import MLkNN  
from sklearn.naive_bayes import GaussianNB
import inspect
class ComplexMul():
  adjList = []
  matrixAdj = []
  graphs = []
  k = None
  n = labels = None
  globalAssortativity = []
  clustering = []
  assortativity = []
  degree = []
  relevantDegree = []
  X_train = []
  y_train = []
  number_edges = []
  classifier = []
  t_classifierProba = []
  delta = None
  lambd = None
  threshold = None
  test_id = []
  proportion = None
  def __init__(self,  k = 5, classifier = MLkNN(), lambd = 0.3, delta = .5, threshold = 0.70):
    self.k = k
    self.classifier = classifier
    self.lambd = lambd
    self.delta = delta
    self.threshold = threshold
    
  def build_graph(self):
    euclidean_dist = euclidean_distances(self.X_train)
    np.fill_diagonal(euclidean_dist, np.inf)
    ind_ranking = np.argsort(euclidean_dist, axis = 1)[:,:self.k]
    for a in range(self.labels):
      edge_list = []
      for b in range(self.n): 
        for c in range(len(ind_ranking[b])):
          if self.y_train[b][a] == 1 and self.y_train[ind_ranking[b][c]][a] == 1:
            edge_list.append([b, ind_ranking[b][c]])
        edge_list.append([b, b]) 
      self.graphs.append(ig.Graph(edge_list, directed = False).simplify())
      self.test_id.append(self.graphs[a].vcount())

  def local_clustering_coefficient(self, graph):
    return ig.Graph.transitivity_local_undirected(graph)

  def fit(self, X_train, y_train):
    n_edges = []
    self.X_train = X_train
    self.y_train = y_train
    self.classifier.fit(X_train, self.y_train)
    self.n, self.labels = X_train.shape[0], y_train.shape[1]
    self.build_graph()
    proportion = [0 for i in range(self.labels)]
    for u in y_train:
      for i in range(len(u)):
        proportion[i] += u[i]
    self.proportion = proportion
    self.matrixAdj = [[[0 for i in range(self.n + 1)] for j in range(self.n + 1)] for jj in range(self.labels)]

    for i in range(self.labels):
      self.clustering.append(ig.Graph.transitivity_avglocal_undirected(self.graphs[i]))
      self.assortativity.append(ig.Graph.assortativity_degree(self.graphs[i]))

    return self

  def get_test_variation(self, test_object):
    distances = []

    for i in range(self.n):
      distances.append(distance.euclidean(self.X_train[i], test_object))
  
    ind_ranking = np.argsort(distances)[:self.k]

    for i in range(self.labels):
      self.graphs[i].add_vertices(1)

    allEdgesToRemove = [0 for i in range(self.labels)]
    for i in range(self.labels):
      for j in range(len(ind_ranking)):
        y = ind_ranking[j]
        if self.y_train[y][i] == 1:
          self.graphs[i].add_edge(self.test_id[i], y)
          allEdgesToRemove[i] += 1
    clus = [0 for i in range(self.labels)]
    assortativity = [0 for i in range(self.labels)]
    for i in range(self.labels):      
      # print(self.clustering[i], ig.Graph.transitivity_avglocal_undirected(self.graphs[i]))
      clus[i] = 1 - (math.fabs((self.clustering[i]) - ig.Graph.transitivity_avglocal_undirected(self.graphs[i])) * self.proportion[i])
      assortativity[i] = 2 - (math.fabs(self.assortativity[i] - ig.Graph.assortativity_degree(self.graphs[i])) * self.proportion[i])

    _max = np.max(assortativity)
    for i in range(self.labels):
      assortativity[i] /= _max
    for i in range(self.labels):
      if allEdgesToRemove[i] == 0:
        clus[i] = 0
        assortativity[i] = 0
    for i in range(self.labels):
      ig.Graph.delete_vertices(self.graphs[i], self.test_id[i])
    return [clus, assortativity]


  def predict(self, X_test):

    self.t_classifierProba = ((self.classifier.predict_proba(X_test))).toarray()
    prediction = []
    k = 0
    for test in X_test:
      prob = self.get_test_variation(test)
      prob_ = 0
      p = []
      for i in range(self.labels):
        _delta = 1 - self.delta
        _lambda = 1 - self.lambd
        prob_ = (self.lambd * ((prob[0][i] * self.delta) + (prob[1][i] * _delta))) + _lambda * self.t_classifierProba[k][i]
        
        p.append(1 if prob_ >= self.threshold else 0)
      prediction.append(p)
      k += 1

    return np.asarray(prediction)


  # def _get_param_names(cls):
  #   init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
  #   if init is object.__init__:
  #     return []

  #   init_signature = inspect.signature(init)
  #   parameters = [p for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]
  #   for p in parameters:
  #     if p.kind == p.VAR_POSITIONAL:
  #       raise RuntimeError('deu ruim')

  #   return sorted([p.name for p in parameters])

  # def set_params(self, **params):
  #   if not params:
  #     return self

  #   valid_params = self.get_params(deep = True)
  #   nested_params = defaultdict(dict)
  #   for key, value in params.items():
  #     key, delim, sub_key = key.partition('__')
  #     if key not in valid_params:
  #       raise ValueError('deu ruim', key)
  #     if delim:
  #       nested_params[key][sub_key] = value
  #     else:
  #       setattr(self, key, value)
  #       valid_params[key] = value

  #   for key, sub_params in nested_params.items():
  #     valid_params[key].set_params(**sub_params)
  #   return self

  # def get_params(self, deep = True):
  #   out = dict()
  #   for key in self._get_param_names():
  #     value = getattr(self, key, None)
  #     if deep and hasattr(value, 'get_params'):
  #       deep_items = value.get_params().items()
  #       out.update((key + '__' + k, val) for k, val in deep_items)
  #     out[key] = value
  #   return out