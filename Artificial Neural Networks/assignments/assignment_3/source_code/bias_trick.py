import numpy as np 
# Data: study hours vs exam score (score = 35 + 6 * hours + noise) 
rng = np.random.default_rng(42) 
hours = rng.uniform(0, 10, 100) 
score = 35 + 6 * hours + rng.normal(0, 4, 100) 

X = np.c_[hours, np.ones(len(hours))]     # each row is [x, 1] 

W = np.zeros(2)                           # W = [w, b] 

alpha, epochs = 0.01, 2643 

for epoch in range(epochs): 

    error = X.dot(W) - score 
    W -= alpha * X.T.dot(error) / len(score) 
    print(error)

loss = np.mean((X.dot(W) - score) ** 2) / 2 
print(f"w = {W[0]:.3f}, b = {W[1]:.3f}, loss = {loss:.3f}") 