#from perceptron import Perceptron
import numpy as np


#Activation function (step)
def step(x):
    return 1 if x > 0 else 0

def fit(X, y, epochs=10):
    # BIAS TRICK: Add a column of one's as a last entry in the
    # feature matrix, 
    global W
    X = np.c_[X, np.ones((X.shape[0]))]
    for epoch in np.arange(0, epochs):
        for (x, target) in zip(X, y):
            # A dot product between the input features 
            p = step(np.dot(x, W))

            if p != target:
                error = p - target
                W += -alpha * error * x
                print(error) 

def predict(X, addBias=True):
    #ensure our input is a matrix
    X = np.atleast_2d(X)

    if addBias:
        X = np.c_[X, np.ones((X.shape[0]))]

    return step(np.dot(X, W))


#### MAIN CODE To RUN ####
#generate dataset for X and labels y
X = np.array([[0,0], [0,1], [1,0], [1,1]])
#OR Dataset
#y = np.array([[0], [1], [1], [1]])
#AND Dataset
#y = np.array([[0], [0], [0], [1]])
#XOR Dataset
y = np.array([[0], [1], [1], [0]])

#print(X.shape)
print("[INFO] Training perceptron...")
#p = Perceptron(X.shape[1], alpha=0.1)
W = np.random.randn(X.shape[1]+1) / np.sqrt(X.shape[1])
alpha = 9.1

fit(X, y, epochs=2000)
print("[INFO] Testing perceptron...")

for (x, target) in zip(X, y):
    #print(x)
    pred = predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(
        x, target[0], pred))



