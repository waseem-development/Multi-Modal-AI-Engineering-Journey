import numpy as np
import tensorflow as tf


class Perceptron:

    def __init__(self, N, alpha=0.1):
        self.W = tf.Variable(
            tf.random.normal([N + 1]) /
            tf.sqrt(tf.cast(N, tf.float32))
        )
        self.alpha = alpha

    def step(self, x):
        return tf.cast(x > 0, tf.float32)

    def fit(self, X, y, epochs=10):
        X = tf.concat(
            [X, tf.ones((X.shape[0], 1))],
            axis=1
        )

        for epoch in range(epochs):
            for x, target in zip(X, y):
                p = self.step(tf.reduce_sum(x * self.W))
                error = p - target

                if p.numpy() != target.numpy():
                    self.W.assign_sub(
                        self.alpha * error * x
                    )

    def predict(self, X, addBias=True):
        X = tf.convert_to_tensor(
            np.atleast_2d(X),
            dtype=tf.float32
        )

        if addBias:
            X = tf.concat(
                [X, tf.ones((X.shape[0], 1))],
                axis=1
            )

        output = tf.linalg.matmul(
            X, tf.reshape(self.W, [-1, 1])
        )

        return self.step(output)


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([0, 1, 1, 1], dtype=np.float32)

X = tf.constant(X)
y = tf.constant(y)

print("[INFO] Training perceptron...")

p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=100)

print("[INFO] Testing perceptron...")

for x, target in zip(X, y):
    pred = p.predict(x)

    print(
        f"data={x.numpy()}, "
        f"ground-truth={target.numpy()}, "
        f"pred={pred.numpy()[0][0]}"
    )
