import MultiLayerPerceptron as mlp
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1, 1, 0]]).T

model = mlp.MultiLayerPerceptron(4, 4, 10000)
model.fit(X, y)

print(model.predict(np.array([0, 0])))