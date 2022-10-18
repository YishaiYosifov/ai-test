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

tf.random.set_seed(1)

def sort_files():
    for name in NAMES:
        os.makedirs(BASE_DIR + "train/" + name)
        os.makedirs(BASE_DIR + "val/" + name)
        os.makedirs(BASE_DIR + "test/" + name)

    originFlders = ["0001/", "0002/", "0003/", "0004/", "0005/"]
    for folderIndex, folder in enumerate(originFlders):
        files = os.listdir(BASE_DIR + folder)

        numberOfImages = len([name for name in files])

        train = int((numberOfImages * 0.6) + 0.5)
        valid = int((numberOfImages * 0.25) + 0.5)
        test = numberOfImages - train - valid

        for index, file in enumerate(files):
            fileName = BASE_DIR + folder + file
            if index < train: shutil.move(fileName, BASE_DIR + "train/" + NAMES[folderIndex])
            elif index < train + valid: shutil.move(fileName, BASE_DIR + "val/" + NAMES[folderIndex])
            else: shutil.move(fileName, BASE_DIR + "test/" + NAMES[folderIndex])

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
    if not os.path.isdir(BASE_DIR + "/train/"): sort_files()

    #trainGenerator = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True,
    #                                    rotation_range=20,
    #                                    width_shift_range=0.2, height_shift_range=0.2,
    #                                    shear_range=0.2, zoom_range=0.2)
    trainGenerator = ImageDataGenerator(rescale=1.0/255)
    validGenerator = ImageDataGenerator(rescale=1.0/255)
    testGenerator = ImageDataGenerator(rescale=1.0/255)

    trainBatches = trainGenerator.flow_from_directory(
        BASE_DIR + "train",
        target_size=(256, 256),
        class_mode="sparse",
        batch_size=4,
        classes=NAMES
    )

    validBatches = validGenerator.flow_from_directory(
        BASE_DIR + "val",
        target_size=(256, 256),
        class_mode="sparse",
        shuffle=False,
        batch_size=4,
        classes=NAMES
    )

    testBatches = testGenerator.flow_from_directory(
        BASE_DIR + "test",
        target_size=(256, 256),
        class_mode="sparse",
        shuffle=False,
        batch_size=4,
        classes=NAMES
    )


    if not os.path.isfile("lego.h5"):
        model = Sequential([
            Conv2D(32, 3, activation="relu", input_shape=(256, 256, 3)),
            MaxPool2D(),
            Conv2D(64, 3, activation="relu"),
            MaxPool2D(),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(128, activation="relu"),
            Dense(5)
        ])

        model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

        earlyStopping = EarlyStopping(patience=7, verbose=1)
        history = model.fit(trainBatches, validation_data=validBatches, callbacks=[earlyStopping], epochs=30, verbose=1)
        model.save("lego.h5")

        pyplot.figure(figsize=(16, 6))
        pyplot.subplot(1, 2, 1)
        pyplot.plot(history.history["loss"], label="train loss")
        pyplot.plot(history.history["val_loss"], label="valid loss")
        pyplot.grid()
        pyplot.legend(fontsize=15)

        pyplot.subplot(1, 2, 2)
        pyplot.plot(history.history["accuracy"], label="train acc")
        pyplot.plot(history.history["val_accuracy"], label="valid acc")
        pyplot.grid()
        pyplot.legend(fontsize=15)
        pyplot.show()
    else: model = load_model("lego.h5")

    model.evaluate(testBatches, verbose=1)

    predictions = model.predict(testBatches)
    predictions = tf.nn.softmax(predictions)
    labels = numpy.argmax(predictions, axis=1)
    print(testBatches[1][1])
    print(labels[4:8])
    show(testBatches[1], labels[4:8])