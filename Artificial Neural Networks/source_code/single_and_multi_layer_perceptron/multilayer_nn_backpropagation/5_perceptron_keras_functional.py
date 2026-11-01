import numpy as np
import tensorflow as tf

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

# OR dataset
y = np.array([[0], [1], [1], [1]], dtype=np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("[INFO] Training perceptron...")
model.fit(X, y, epochs=200, verbose=0)

print("[INFO] Testing perceptron...")

predictions = model.predict(X, verbose=0)

for x, target, pred in zip(X, y, predictions):
    print(
        f"data={x}, ground-truth={target[0]}, "
        f"pred={int(pred[0] >= 0.5)}"
    )
