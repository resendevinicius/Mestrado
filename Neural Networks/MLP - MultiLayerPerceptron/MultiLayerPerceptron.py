import numpy as np
import random

class HiddenLayer():
  weights = []
  layers = []
  def __init__(self, number_neurons, number_input_neurons):
    self.weights = 2 * np.random.random((number_input_neurons, number_neurons)) - 1 
    self.layers = np.full((number_input_neurons, number_neurons), 0)

  def get_weights(self):
    return self.weights
  def get_layers(self):
    return self.layers

class MultiLayerPerceptron():
  n_layers = 0
  cycles = 0
  n_hiddenneurons = 0
  weights = []
  bias = []
  outputfromlayer = []
  output = []
  X = []
  y = []
  def __init__(self, n_layers, n_hiddenneurons, cycles):
    self.n_layers = n_layers
    self.cycles = cycles
    self.n_hiddenneurons = n_hiddenneurons


  def act_function(self, x, option):
    if option == 1:
      return 1 / (1 + np.exp(-x))
    else:
      return x * (1 - x)

  def feed_forward(self):
    self.outputfromlayer[0] = self.act_function(np.dot(self.X, self.weights[0]), 1)
    for i in range(1, self.n_layers):
      self.outputfromlayer[i] = self.act_function(np.dot(self.outputfromlayer[i - 1], self.weights[i]), 1)

  def back_propagation(self):
    layerserrors = [0 for i in range(self.n_layers)]
    layersdelta = [0 for i in range(self.n_layers)]
    layersadjust = [0 for i in range(self.n_layers)]
    layerserrors[self.n_layers - 1] = self.y - self.outputfromlayer[self.n_layers - 1]
    layersdelta[self.n_layers - 1] = layerserrors[self.n_layers - 1] * self.act_function(self.outputfromlayer[self.n_layers - 1], 2)
    for i in range(self.n_layers - 2, -1, -1):
      layerserrors[i] = np.dot(layersdelta[i + 1], self.weights[i + 1].T)
      layersdelta[i] = layerserrors[i] * self.act_function(self.outputfromlayer[i], 2)

    layersadjust[0] = self.X.T.dot(layersdelta[0])
    for i in range(1, self.n_layers):
      layersadjust[i] = self.outputfromlayer[i - 1].T.dot(layersdelta[i])
      self.weights[i] += layersadjust[i]
    self.weights[0] += layersadjust[0]

  def fit(self, X, y):
    self.X = X
    self.y = y
    self.n_layers -= 1
    self.weights.append(HiddenLayer(self.n_hiddenneurons, len(X[0])).get_weights())
    self.outputfromlayer.append(HiddenLayer(self.n_hiddenneurons, len(X[0])).get_layers())
    for i in range(self.n_layers - 2):
      self.weights.append(HiddenLayer(self.n_hiddenneurons, self.n_hiddenneurons).get_weights())
      self.outputfromlayer.append(HiddenLayer(self.n_hiddenneurons, self.n_hiddenneurons).get_layers())
    self.weights.append(HiddenLayer(1, self.n_hiddenneurons).get_weights())
    self.outputfromlayer.append(HiddenLayer(1, self.n_hiddenneurons).get_layers())
    
    self.bias = [[1 - random.random() for i in range(self.n_hiddenneurons)] for j in range(self.n_layers)]
    for i in range(self.cycles):
      self.feed_forward()
      self.back_propagation()

  def predict(self, test):
    outputfromlayer = []
    outputfromlayer.append(self.act_function(np.dot(test, self.weights[0]), 1))
    for i in range(1, self.n_layers):
      outputfromlayer.append(self.act_function(np.dot(outputfromlayer[i - 1], self.weights[i]), 1))

    return outputfromlayer[len(outputfromlayer) - 1]
