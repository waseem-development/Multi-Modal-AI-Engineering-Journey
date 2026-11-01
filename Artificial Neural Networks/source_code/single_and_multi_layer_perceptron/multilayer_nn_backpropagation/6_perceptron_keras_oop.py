import numpy as np
import tensorflow as tf


class Perceptron:

    def __init__(self, N, alpha=0.1):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(N,)),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        self.alpha = alpha

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=self.alpha
            ),
            loss="binary_crossentropy"
        )

    def fit(self, X, y, epochs=10):
        self.model.fit(X, y, epochs=epochs, verbose=0)

    def predict(self, X):
        X = np.atleast_2d(X)
        predictions = self.model.predict(X, verbose=0)
        return (predictions >= 0.5).astype(int)


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([[0], [1], [1], [1]], dtype=np.float32)

print("[INFO] Training perceptron...")

p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=200)

print("[INFO] Testing perceptron...")

for x, target in zip(X, y):
    pred = p.predict(x)

    print(
        f"data={x}, ground-truth={target[0]}, "
        f"pred={pred[0][0]}"
    )
