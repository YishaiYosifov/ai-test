from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Flatten, Dense, Softmax
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.datasets import mnist
from matplotlib import pyplot

import tensorflow as tf
import numpy
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# normalize: 0, 255 -> 0, 1
xTrain, xTest = xTrain / 255, xTest / 255

# model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(10)
])


# loss and optimzer
loss = SparseCategoricalCrossentropy(from_logits=True)
optimizer = Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# training
model.fit(xTrain, yTrain, batch_size=64, epochs=5, shuffle=True, verbose=1)

# evaluate
model.evaluate(xTest, yTest, batch_size=64, verbose=1)

# predictions
probabilityModel = Sequential([
    model,
    Softmax()
])

model.save("model")

newModel = load_model("model")
model.evaluate(xTest, yTest, batch_size=64, verbose=1)

model.save_weights("model_weights")
model.load_weights("model_weights")

architecture = model.to_json()
with open("architecture.json", "w") as f: json.dump(architecture, f)

with open("architecture.json", "r") as f: architecture = json.load(f)