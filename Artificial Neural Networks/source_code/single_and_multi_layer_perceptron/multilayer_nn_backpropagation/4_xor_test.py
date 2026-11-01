#from neuralnet import NeuralNetwork 
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=200)

#W = []
#layers = [2,2,1] #layers
#alpha = 0.1 #alpha
