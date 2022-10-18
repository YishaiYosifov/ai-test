from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from matplotlib import pyplot
from keras.layers import *

import tensorflow as tf
import shutil
import random
import pandas
import keras
import numpy
import math
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = "star-wars-images/"
NAMES = ["YODA", "LUKE SKYWALKER", "R2-D2", "MACE WINDU", "GENERAL GRIEVOUS"]

def show(batch, pred_labels=None):
    pyplot.figure(figsize=(10,10))
    for i in range(4):
        pyplot.subplot(2,2,i+1)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.grid(False)
        pyplot.imshow(batch[0][i], cmap=pyplot.cm.binary)
        lbl = NAMES[int(batch[1][i])]
        if pred_labels is not None:
            lbl += "/ Pred:" + NAMES[int(pred_labels[i])]
        pyplot.xlabel(lbl)
    pyplot.show()

if __name__ == "__main__":
    vggModel = VGG16()
    model = Sequential()
    for layer in vggModel.layers[:-1]:
        layer.trainable = False
        model.add(layer)

    model.add(Dense(5))

    model.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    trainGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)
    validGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)
    testGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)

    trainBatches = trainGenerator.flow_from_directory(
        BASE_DIR + "train",
        target_size=(224, 224),
        class_mode="sparse",
        batch_size=4,
        classes=NAMES
    )

    validBatches = validGenerator.flow_from_directory(
        BASE_DIR + "val",
        target_size=(224, 224),
        class_mode="sparse",
        shuffle=False,
        batch_size=4,
        classes=NAMES
    )

    testBatches = testGenerator.flow_from_directory(
        BASE_DIR + "test",
        target_size=(224, 224),
        class_mode="sparse",
        shuffle=False,
        batch_size=4,
        classes=NAMES
    )

    if not os.path.isfile("lego.h5"):
        earlyStopping = EarlyStopping(patience=7, verbose=1)
        history = model.fit(trainBatches, validation_data=validBatches, callbacks=[earlyStopping], epochs=30, verbose=1)
        model.save("lego.h5")
    else: model = load_model("lego.h5")

    model.evaluate(testBatches, verbose=1)

    predictions = model.predict(testBatches)
    predictions = tf.nn.softmax(predictions)
    labels = numpy.argmax(predictions, axis=1)
    print(testBatches[0][1])
    print(labels[:4])
    show(testBatches[0], labels[:4])