import numpy as np 
# Data: study hours vs exam score (score = 35 + 6 * hours + noise) 

rng = np.random.default_rng(42) 
hours = rng.uniform(0, 10, 100) 
score = 35 + 6 * hours + rng.normal(0, 4, 100) 

X = np.c_[hours, np.ones(len(hours))]     # bias trick
W = np.zeros(2) 
alpha, epochs, batch_size = 0.01, 1000, 20 

for epoch in range(epochs): 

    idx = rng.permutation(len(score))     # shuffle every epoch 

    X, score = X[idx], score[idx] 

    for i in range(0, len(score), batch_size): 

        bx, by = X[i:i + batch_size], score[i:i + batch_size] 

        W -= alpha * bx.T.dot(bx.dot(W) - by) / len(by)   # update per batch 


loss = np.mean((X.dot(W) - score) ** 2) / 2 
print(f"w = {W[0]:.3f}, b = {W[1]:.3f}, loss = {loss:.3f}") 
print(f"updates per epoch = {len(score) // batch_size}") 