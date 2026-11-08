import numpy as np


def step(x):
    return 1 if x > 0 else 0

def fit(X, y, epochs=100, alpha=0.1):
    global W

    X = np.c_[X, np.ones(len(X))]

    for epoch in range(epochs):
        for x, target in zip(X, y):
            pred = step(np.dot(x, W))
            error = target - pred
            W += alpha * error * x

def predict(X):
    X = np.atleast_2d(X)
    X = np.c_[X, np.ones(len(X))]
    return [step(np.dot(x, W)) for x in X]

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

and_ = np.array([0, 0, 0, 1])
or_ = np.array([0, 1, 1, 1])
xor_ = np.array([0, 1, 1, 0])

y = or_  

W = np.random.randn(3) * 0.1

print("Training perceptron...")
fit(X, y, epochs=100)

print("Testing perceptron...")

for x, target, pred in zip(X, y, predict(X)):
    print(f"Input={x}, Actual={target}, Predicted={pred}")