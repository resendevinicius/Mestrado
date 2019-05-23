# Author: Vinicius Resende 
# 15/05/2019 - 20:05
# resendeviniciush@gmail.com

import numpy as np
import random
import math


class HiddenLayer():

  weights = []
  layers = []

  def __init__(self, number_neurons, number_input_neurons):
    self.weights = 1 - np.random.random((number_input_neurons, number_neurons))
    self.layers = np.full((number_input_neurons, number_neurons), 0)

  def get_weights(self):
    return self.weights

  def get_layers(self):
    return self.layers

class MultiLayerPerceptron():

  n_layers = 0
  cycles = 0
  n_neurons = 0
  weights = []
  bias = []
  outputfromlayer = []
  output = []
  X = []
  y = []
  alpha = 0
  outputbias = 0
  def __init__(self, n_layers, n_neurons, cycles):

    self.n_layers = n_layers
    self.cycles = cycles
    self.n_neurons = n_neurons


  def act_function(self, x, option):

    if option == 1:
      return 1 / (1 + np.exp(-x))
    else:
      return x * (1 - x)

  def feed_forward(self, X):

    out = np.dot(X, self.weights[0]) + np.transpose(self.bias[0])
    self.outputfromlayer[0] = self.act_function(out, 1)
  
    for i in range(1, self.n_layers - 1):
      out = np.dot(self.outputfromlayer[i - 1], self.weights[i]) + np.transpose(self.bias[i])
      self.outputfromlayer[i] = self.act_function(out, 1)

    out = np.dot(self.outputfromlayer[self.n_layers - 2], self.weights[self.n_layers - 1])
    out += self.outputbias
    self.outputfromlayer[self.n_layers - 1] = self.act_function(out, 1)

  def back_propagation(self, X, y):

    layerserrors = [0 for i in range(self.n_layers)]
    layersdelta = [0 for i in range(self.n_layers)]
    layersadjust = [0 for i in range(self.n_layers)]

    layerserrors[self.n_layers - 1] = ((self.outputfromlayer[self.n_layers - 1] - y))
    layersdelta[self.n_layers - 1] =  (layerserrors[self.n_layers - 1] * self.act_function(self.outputfromlayer[self.n_layers - 1], 2))
    
    for i in range(self.n_layers - 2, -1, -1):
      layerserrors[i] = np.dot(layersdelta[i + 1], self.weights[i + 1].T)
      layersdelta[i] = (layerserrors[i] * self.act_function(self.outputfromlayer[i], 2))
    

    layersadjust[0] = self.alpha * (X.T.dot(layersdelta[0]))

    for i in range(1, self.n_layers):
      print((self.outputfromlayer[i - 1]))
      print((layersdelta[i]))
      layersadjust[i] = self.alpha * self.outputfromlayer[i - 1].T.dot(layersdelta[i])
      if i < self.n_layers - 1:
        self.bias[i] -= (self.alpha * layersdelta[i])
      self.weights[i] -= (layersadjust[i])

    self.outputbias = (self.alpha * layersdelta[self.n_layers - 1])
    self.bias[0] -= (self.alpha * layersdelta[0])
    self.weights[0] -= (layersadjust[0])
  

  def fit(self, X, y, alpha = 0.0001):

    self.X = X
    self.y = y
    self.n_layers -= 1
    self.alpha = alpha
    self.weights.append(HiddenLayer(self.n_neurons, len(X[0])).get_weights())
    self.outputfromlayer.append(HiddenLayer(self.n_neurons, len(X[0])).get_layers())

    for i in range(self.n_layers - 2):
      self.weights.append(HiddenLayer(self.n_neurons, self.n_neurons).get_weights())
      self.outputfromlayer.append(HiddenLayer(self.n_neurons, self.n_neurons).get_layers())

    self.weights.append(HiddenLayer(1, self.n_neurons).get_weights())
    self.outputfromlayer.append(HiddenLayer(1, self.n_neurons).get_layers())
    self.bias = [[1 - random.random() for i in range(self.n_neurons)] for j in range(self.n_layers)]
    self.outputbias = 1 - random.random()
    for i in range(self.cycles):
      for j in range(len(X)):
        self.feed_forward(X[j])
        self.back_propagation(X[j], y[j])

  def predict(self, test):
    outputfromlayer = []
    outputfromlayer.append(self.act_function(np.dot(test, self.weights[0]) + np.transpose(self.bias[0]), 1))
    for i in range(1, self.n_layers):
      outputfromlayer.append(self.act_function(np.dot(self.outputfromlayer[i - 1], self.weights[i]) + np.transpose(self.bias[i]), 1))
    return outputfromlayer[len(outputfromlayer) - 1]
