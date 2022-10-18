from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from matplotlib import pyplot
from keras.layers import *

import tensorflow as tf
import keras
import numpy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# normalize: 0, 255 -> 0, 1
xTrain, xTest = xTrain / 255, xTest / 255

# model
#model = Sequential([
#    keras.Input((28, 28)),
#    SimpleRNN(128, return_sequences=True, activation="relu"),
#    SimpleRNN(128, return_sequences=False, activation="relu"),
#    Dense(10)
#])

model = Sequential([
    keras.Input((28, 28)),
    GRU(128, return_sequences=True, activation="relu"),
    GRU(128, return_sequences=False, activation="relu"),
    Dense(10)
])

#model = Sequential([
#    keras.Input((28, 28)),
#    LSTM(128, return_sequences=True, activation="relu"),
#    LSTM(128, return_sequences=False, activation="relu"),
#    Dense(10)
#])

# loss and optimzer
loss = SparseCategoricalCrossentropy(from_logits=True)
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# training
model.fit(xTrain, yTrain, batch_size=64, epochs=5, shuffle=True, verbose=1)

# evaluate
model.evaluate(xTest, yTest, batch_size=64, verbose=1)