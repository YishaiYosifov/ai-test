from keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.layers import *

import keras
import numpy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

inputs = keras.Input(shape=(28, 28))
flatten = Flatten()
dense1 = Dense(128, activation="relu")

outputDense1 = Dense(10, activation="softmax", name="category_output")
outputDense2 = Dense(1, activation="sigmoid", name="leftright_output")

x = flatten(inputs)
x = dense1(x)
outputs1 = outputDense1(x)
outputs2 = outputDense2(x)

model = keras.Model(inputs=inputs, outputs=[outputs1, outputs2], name="mnist_model")

losses = {
    "category_output": SparseCategoricalCrossentropy(from_logits=False),
    "leftright_output": BinaryCrossentropy(from_logits=False)
}

model.compile(loss=losses, optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTrain, xTest = xTrain / 255.0, xTest / 255.0

yLeftRight = numpy.zeros(yTrain.shape, dtype=numpy.uint8)
for index, y in enumerate(yTrain):
    if y > 5: yLeftRight[index] = 1

y = {
    "category_output": yTrain,
    "leftright_output": yLeftRight
}

model.fit(xTrain, y=y, epochs=5, batch_size=64, verbose=1)

predictions = model.predict(xTest)

predictionCategory = predictions[0]
predictionLeftRight = predictions[1]

labelsCategory = numpy.argmax(predictionCategory, axis=1)
labelsLeftRight = numpy.array([numpy.round(prediction) for prediction in predictionLeftRight], dtype=numpy.int32)

print(yTest[:20])
print(labelsCategory[:20])
print(labelsLeftRight[:20].reshape(-1))