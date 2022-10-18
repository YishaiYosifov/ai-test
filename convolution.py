from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential, load_model
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.layers import *

import tensorflow as tf
import numpy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
numpy.set_printoptions(precision=3, suppress=True)

(trainImages, trainLabels), (testImages, testLabels) = cifar10.load_data()

trainImages, testImages = trainImages / 255, testImages / 255

classNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

if not os.path.isfile("image.h5"):
    model = Sequential([
        Conv2D(32, 3, activation="relu", input_shape=(32, 32, 3)),
        MaxPool2D(),
        Conv2D(32, 3, activation="relu"),
        MaxPool2D(),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(10)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(trainImages, trainLabels, epochs=10, batch_size=5, verbose=1)
    model.save("image.h5")
else: model = load_model("image.h5")

model.evaluate(testImages, testLabels, batch_size=5, verbose=2)
predictions = model.predict(testImages)
predictions = [numpy.round(prediction[0]) for prediction in predictions]