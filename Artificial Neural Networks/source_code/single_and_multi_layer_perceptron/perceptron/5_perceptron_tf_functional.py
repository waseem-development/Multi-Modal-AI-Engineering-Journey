import numpy as np
import tensorflow as tf


def step(x):
    return tf.cast(x > 0, tf.float32)


def fit(X, y, epochs=10):
    global W

    X = tf.concat(
        [X, tf.ones((X.shape[0], 1))],
        axis=1
    )

    for epoch in range(epochs):
        for x, target in zip(X, y):
            p = step(tf.reduce_sum(x * W))
            error = p - target
            W.assign_sub(alpha * error * x)


def predict(X, addBias=True):
    X = tf.convert_to_tensor(
        np.atleast_2d(X),
        dtype=tf.float32
    )

    if addBias:
        X = tf.concat(
            [X, tf.ones((X.shape[0], 1))],
            axis=1
        )

    return step(
        tf.linalg.matmul(X, tf.reshape(W, [-1, 1]))
    )


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([0, 1, 1, 1], dtype=np.float32)

X = tf.constant(X)
y = tf.constant(y)

W = tf.Variable(
    tf.random.normal([X.shape[1] + 1]) /
    tf.sqrt(tf.cast(X.shape[1], tf.float32))
)

alpha = 0.1

print("[INFO] Training perceptron...")
fit(X, y, epochs=100)

print("[INFO] Testing perceptron...")

for x, target in zip(X, y):
    pred = predict(x)

    print(
        f"data={x.numpy()}, "
        f"ground-truth={target.numpy()}, "
        f"pred={pred.numpy()[0][0]}"
    )
