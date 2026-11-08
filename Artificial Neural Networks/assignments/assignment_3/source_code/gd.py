import numpy as np 
# Data: study hours vs exam score (score = 35 + 6 * hours + noise) 
rng = np.random.default_rng(42) 
hours = rng.uniform(0, 10, 100) 

score = 35 + 6 * hours + rng.normal(0, 4, 100) 
w, b = 0.0, 0.0 
alpha, epochs = 0.01, 2643

for epoch in range(epochs): 

    error = (w * hours + b) - score 
    w -= alpha * np.mean(error * hours)   # gradient for w 
    b -= alpha * np.mean(error)           # gradient for b 
    print(f"Error: {error}")

loss = np.mean(((w * hours + b) - score) ** 2) / 2 
print(f"w = {w:.3f}, b = {b:.3f}, loss = {loss:.3f}") 