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
  proportion = None
  def __init__(self, k = 5, classifier = MLkNN(), lambd = 0.3, delta = .5, threshold = 0.70):
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
      self.adjList.append([])
      for b in range(self.n + 1):
        self.adjList[a].append([])

    self.matrixAdj = [[[0 for i in range(self.n + 1)] for j in range(self.n + 1)] for k in range(self.labels)]
    for a in range(self.labels):
      for b in range(self.n): 
        for c in ind_ranking[b]:
          if self.y_train[b][a] == 1 and self.y_train[c][a] == 1:
            self.matrixAdj[a][b][c] = self.matrixAdj[a][c][b] = 1
            self.adjList[a][b].append(c)
            self.adjList[a][c].append(b)
    
    self.number_edges = [0 for i in range(self.labels)]
    self.degree = [[0 for i in range(self.n + 1)] for j in range(self.labels)]
    for a in range(self.labels):
      for b in range(self.n):
        self.adjList[a][b] = list(np.unique(self.adjList[a][b]))
        self.number_edges[a] += len(self.adjList[a][b])
        self.degree[a][b] = len(self.adjList[a][b])

  def local_clustering_coefficient(self, graph):
    return ig.Graph.transitivity_local_undirected(graph)


  def global_clustering_coefficient(self, graph):
    c = 0
    for i in range(len(self.adjList[graph])):
      c += self.vertex_clus_contribution(graph, i)
    return c

  def global_assortativity(self, graph):
    a = b = c = 0

    for i in range(len(graph)):
      x, y, z = self.vertex_assortativity_contribution(graph, i)
      a += x
      b += y
      c += z

    return [a, b, c]
  def average_degree(self, graph):
    return ig.Graph.average_degree(graph)


  def vertex_assortativity_contribution(self, graph, u):
    a = b = c = 0
    for v in graph[u]:
      a += (len(graph[u]) * len(graph[v]))
      b += ((len(graph[u]) + len(graph[v])) / 2.0)
      c += (((len(graph[u]) ** 2) + (len(graph[v]) ** 2)) / 2.0)

    return [a, b, c]

  def vertex_clus_contribution(self, graph, u):
    c = 0

    for v in self.adjList[graph][u]:
      for y in self.adjList[graph][v]:
        if self.matrixAdj[graph][u][y] == 1:
          c += 1

    if c > 1:
      return c / (len(self.adjList[graph][u]) * (len(self.adjList[graph][u]) - 1)) 

    return 0

  def remove_assortativity(self, graph, vertices):
    visited = []
    a, b, c = self.assortativity[graph][:]

    for u in vertices:
      visited.append(u)
      for v in self.adjList[graph][u]:
        if v not in visited:
          a -= 2 * (self.degree[graph][v] * self.degree[graph][u])
          b -= (self.degree[graph][v] + self.degree[graph][u])
          c -= ((self.degree[graph][v] ** 2) + (self.degree[graph][u] ** 2))

    return [a, b, c]
  def fit(self, X_train, y_train):

    self.X_train = X_train
    self.y_train = y_train
    self.classifier.fit(X_train, self.y_train)
    self.n, self.labels = X_train.shape[0], y_train.shape[1]
    self.adjList.clear()
    self.build_graph()
    proportion = [0 for i in range(self.labels)]
    for u in y_train:
      for i in range(len(u)):
        proportion[i] += u[i]
    self.proportion = proportion

    for i in range(self.labels):
      self.clustering.append(self.global_clustering_coefficient(i))
      self.assortativity.append(self.global_assortativity(self.adjList[i]))
      relevantDegree = 0
      for j in range(len(self.adjList[i])):
        if self.degree[i][j] > 1:
          relevantDegree += 1
      self.relevantDegree.append(relevantDegree)
    self.degree = np.asarray(self.degree)

    A = []
    B = []
    C = []
    
    for g in range(self.labels):
      a = b = c = 0
      for i in range(len(self.adjList[g])):
        if len(self.adjList[g][i]) > 0:
          x, y, z = self.vertex_assortativity_contribution(self.adjList[g], i)
          a += x
          b += y
          c += z
      A.append(a)
      B.append(b)
      C.append(c)


    for i in range(self.labels):
      newA = A[i] 
      newB = B[i]
      newC = C[i]
      newA /= self.number_edges[i]
      newB /= self.number_edges[i]
      newB = newB ** 2
      newC /= self.number_edges[i]
      self.globalAssortativity.append((newA - newB) / (newC - newB))

    return self

  def add_test_assortativity(self, graph, test, number_edges, assortativity):
    a, b, c = assortativity
    test_degree = len(self.adjList[graph][test])
    for u in self.adjList[graph][test]:
      adjDegree = len(self.adjList[graph][u])
      a += 2 * test_degree * adjDegree
      b += test_degree + adjDegree
      c += (test_degree ** 2) + (adjDegree ** 2)

    visited = []
    for u in self.adjList[graph][test]:
      visited.append(u)
      adjDegree = len(self.adjList[graph][u])
      for v in self.adjList[graph][u]:
        if v not in visited and test != v:
          adjDegree_2 = len(self.adjList[graph][v])
          a += 2 * adjDegree_2 * adjDegree
          b += adjDegree_2 + adjDegree
          c += (adjDegree_2 ** 2) + (adjDegree ** 2)

    a /= number_edges
    b /= number_edges
    b *= b
    c /= number_edges

    return ((a - b) / (c - b))

  def get_test_variation(self, test_object):
    distances = []

    for i in range(self.n):
      distances.append(distance.euclidean(self.X_train[i], test_object))
  
    ind_ranking = np.argsort(distances)[:self.k]

    clus = self.clustering[:]
    number_edges = self.number_edges[:]
    
    assortativity = []

    for i in range(self.labels):
      for v in ind_ranking:
        if self.degree[i][v] > 1:
          clus[i] -= self.vertex_clus_contribution(i, v)
      assortativity.append(self.remove_assortativity(i, ind_ranking))  
 
    
    relevantDegree = self.relevantDegree[:]
    allEdgesToRemove = []
    for i in range(self.labels):
      edgesToRemove = []
      for y in ind_ranking:
        if self.y_train[y][i] == 1:
          self.adjList[i][y].append(self.n)
          self.adjList[i][self.n].append(y)
          self.matrixAdj[i][self.n][y] = self.matrixAdj[i][y][self.n] = 1
          number_edges[i] += 2
          edgesToRemove.append(y)
          if(self.degree[i][y] == 1):
            relevantDegree[i] += 1
      relevantDegree[i] += 1 if len(edgesToRemove) > 1 else 0
      allEdgesToRemove.append(edgesToRemove)
      


    for i in range(self.labels):
      for v in ind_ranking:
        clus[i] += self.vertex_clus_contribution(i, v)
      clus[i] += self.vertex_clus_contribution(i, self.n)    
      assortativity[i] = (self.add_test_assortativity(i, self.n, number_edges[i], assortativity[i]))
      clus[i] /= relevantDegree[i]
      clus[i] = 1 - (math.fabs((self.clustering[i] / self.relevantDegree[i]) - clus[i]) * self.proportion[i])
      assortativity[i] = 2 - (math.fabs(assortativity[i] - self.globalAssortativity[i]) * self.proportion[i])

    _max = np.max(assortativity)
    for i in range(self.labels):
      assortativity[i] /= _max
    for i in range(len(allEdgesToRemove)):
      if len(allEdgesToRemove[i]) == 0:
        clus[i] = 0
        assortativity[i] = 0
    for i in range(self.labels):
      for element in allEdgesToRemove[i]:
        self.matrixAdj[i][element][self.n] = self.matrixAdj[i][self.n][element] = 0
        self.adjList[i][element].pop()
      self.adjList[i][self.n].clear()

    
    return [clus, assortativity]


  def predict(self, X_test):

    self.t_classifierProba = self.classifier.predict_proba(X_test).toarray()
    
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