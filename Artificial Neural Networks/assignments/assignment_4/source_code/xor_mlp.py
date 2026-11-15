import numpy as np

# XOR data (the last column of 1s is the bias trick)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(1)
W1 = np.random.randn(3, 4)       # input -> hidden (4 nodes)
W2 = np.random.randn(4, 1)       # hidden -> output
alpha = 0.5

for epoch in range(1, 10001):
    # forward pass
    hidden = sigmoid(X.dot(W1))
    output = sigmoid(hidden.dot(W2))

    # backward pass
    d_output = (output - y) * output * (1 - output)
    d_hidden = d_output.dot(W2.T) * hidden * (1 - hidden)

    # update weights
    W2 -= alpha * hidden.T.dot(d_output)
    W1 -= alpha * X.T.dot(d_hidden)

    if epoch % 2000 == 0:
        print(f"Epoch {epoch} | loss = {np.mean((output - y) ** 2) / 2:.5f}")

print("Output:", output.round(3).T, " Expected: [0 1 1 0]")