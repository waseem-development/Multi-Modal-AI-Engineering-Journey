import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# MNIST sample (8x8 digit images), pixels scaled to 0-1, bias trick added
digits = datasets.load_digits()
data = np.c_[digits.data / 16.0, np.ones(len(digits.data))]
labels = np.eye(10)[digits.target]
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(1)
W1 = np.random.randn(65, 32) * 0.1    # input -> hidden (32 nodes)
W2 = np.random.randn(32, 10) * 0.1    # hidden -> output (10 digits)
alpha, n = 2.0, len(trainX)





for epoch in range(1, 1001):
    # forward pass
    hidden = sigmoid(trainX.dot(W1))
    output = sigmoid(hidden.dot(W2))

    # backward pass
    d_output = (output - trainY) * output * (1 - output)
    d_hidden = d_output.dot(W2.T) * hidden * (1 - hidden)

    # update weights
    W2 -= alpha * hidden.T.dot(d_output) / n
    W1 -= alpha * trainX.T.dot(d_hidden) / n

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | loss = {np.mean((output - trainY) ** 2) / 2:.5f}")

# test the network
predictions = sigmoid(sigmoid(testX.dot(W1)).dot(W2)).argmax(axis=1)
print(f"Test accuracy = {np.mean(predictions == testY.argmax(axis=1)) * 100:.2f}%")