from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
from nltk.corpus import stopwords
from keras.optimizers import Adam
from collections import Counter
from matplotlib import pyplot
from keras.layers import *

import tensorflow as tf
import pandas
import string
import keras
import numpy
import nltk
import re
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def remove_url(text : str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", r"", text)

def remove_punctuation(text : str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

def remove_stopwords(text : str) -> str:
    filteredWords = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filteredWords)

def count_unique_words(textColumn) -> Counter:
    counter = Counter()
    for line in textColumn.values:
        for word in line.split(): counter[word] += 1
    return counter

if __name__ == "__main__":
    dataframe = pandas.read_csv("tweets/train.csv")
    dataframe["text"] = dataframe.text.map(remove_url)
    dataframe["text"] = dataframe.text.map(remove_punctuation)

    nltk.download("stopwords", quiet=True)

    stop = set(stopwords.words("english"))

    dataframe["text"] = dataframe.text.map(remove_stopwords)

    numberOfWords = len(count_unique_words(dataframe.text))

    trainSize = int(dataframe.shape[0] * 0.8)
    trainDataframe = dataframe[:trainSize]
    validationDataframe = dataframe[trainSize:]

    trainSentences = trainDataframe.text.to_numpy()
    trainLabels = trainDataframe.target.to_numpy()
    validationSentences = validationDataframe.text.to_numpy()
    validationLabels = validationDataframe.target.to_numpy()

    tokenizer = Tokenizer(num_words=numberOfWords)
    tokenizer.fit_on_texts(trainSentences)
    trainSequences = tokenizer.texts_to_sequences(trainSentences)
    validationSequences = tokenizer.texts_to_sequences(validationSentences)

    maxLength = 20
    trainPadded = pad_sequences(trainSequences, maxlen=maxLength, padding="post", truncating="post")
    validationPadded = pad_sequences(validationSequences, maxlen=maxLength, padding="post", truncating="post")

    model = Sequential([
        Embedding(numberOfWords, 64, input_length=maxLength),
        GRU(64, return_sequences=True),
        GRU(64),
        Dense(1)
    ])

    model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
    model.fit(trainPadded, trainLabels, epochs=5, verbose=1, validation_data=(validationPadded, validationLabels))

    predictions = model.predict(validationPadded)
    predictions = numpy.asarray([1 if prediction > 0.5 else 0 for prediction in predictions], dtype=numpy.int32)

    print(validationSentences[10:20])
    print(validationLabels[10:20])
    print(predictions[10:20])