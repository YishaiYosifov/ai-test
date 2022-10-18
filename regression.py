from keras.layers.preprocessing import normalization
from keras.losses import MeanAbsoluteError
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from matplotlib import pyplot

import tensorflow as tf
import pandas
import numpy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
numpy.set_printoptions(precision=3, suppress=True)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columnNames = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]
dataset = pandas.read_csv(url, names=columnNames, na_values="?", comment="\t", sep=" ", skipinitialspace=True)

dataset = dataset.dropna()
origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1
dataset["Europe"] = (origin == 2) * 1
dataset["Japan"] = (origin == 3) * 1

trainDataset = dataset.sample(frac=0.8, random_state=0)
testDataset = dataset.drop(trainDataset.index)

trainFeatures = trainDataset.copy()
testFeatures = testDataset.copy()
trainLabels = trainFeatures.pop("MPG")
testLabels = testFeatures.pop("MPG")

# Normalization layer
normlizer = normalization.Normalization()

# Adapt to the data
normlizer.adapt(numpy.array(trainFeatures))


feature = "Horsepower"
singleFeature = numpy.array(trainFeatures[feature])

# Normalizatio
singleFeatureNormalizer = normalization.Normalization(input_shape=[1], axis=None)

# adapt to the data
singleFeatureNormalizer.adapt(singleFeature)

singleFeatureModel = Sequential([
    singleFeatureNormalizer,
    Dense(units=64, activation="relu"),
    Dense(units=64, activation="relu"),
    Dense(units=1)
])

loss = MeanAbsoluteError()
optimizer = Adam(learning_rate=0.01)
singleFeatureModel.compile(optimizer=optimizer, loss=loss)

history = singleFeatureModel.fit(trainFeatures[feature], trainLabels, epochs=100, verbose=1, validation_split=0.2)

def plot(feature, x=None, y=None):
    pyplot.figure(figsize=(10, 8))
    pyplot.scatter(trainFeatures[feature], trainLabels, label='Data')
    if x is not None and y is not None: pyplot.plot(x, y, color='k', label='Predictions')
    pyplot.xlabel(feature)
    pyplot.ylabel('MPG')
    pyplot.legend()
    pyplot.show()

singleFeatureModel.evaluate(testFeatures[feature], testLabels, verbose=1)

rangeMin = numpy.min(testFeatures[feature]) - 10
rangeMax = numpy.max(testFeatures[feature]) + 10
x = tf.linspace(rangeMin, rangeMax, 200)
y = singleFeatureModel.predict(x)
plot(feature, x, y)

featureModel = Sequential([
    normlizer,
    Dense(units=64, activation="relu"),
    Dense(units=64, activation="relu"),
    Dense(units=1)
])
featureModel.compile(optimizer=Adam(learning_rate=0.1), loss=loss)
featureModel.fit(trainFeatures, trainLabels, epochs=100, verbose=1, validation_split=0.2)

featureModel.evaluate(testFeatures, testLabels, verbose=1)

rangeMin = numpy.min(testFeatures[feature]) - 10
rangeMax = numpy.max(testFeatures[feature]) + 10
x = tf.linspace(rangeMin, rangeMax, 200)
y = singleFeatureModel.predict(x)
plot(feature, x, y)