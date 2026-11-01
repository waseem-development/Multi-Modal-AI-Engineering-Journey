from neuralnet import NeuralNetwork 
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=3000)

for (x, target) in zip(X, y):
#    print(nn.predict(x)[0])
    pred = nn.predict(x)[0][0]
    step=1 if pred > 0.5 else 0
    print("[INFO] data={}, ground_truth={}, pred={:.4f},step={}".format(x,
        target[0], pred, step))

#W = []
#layers = [2,2,1] #layers
#alpha = 0.1 #alpha
#
#for i in np.arange(0, len(layers)-2):
#    w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
#    W.append( w / np.sqrt(layers[i]))
#    #print(i)
#
##print(w)
## The last two layers are handled separately 
#w = np.random.randn(layers[-2] + 1, layers[-1])
#W.append(w / np.sqrt(layers[-2]))
##print(w)
#
