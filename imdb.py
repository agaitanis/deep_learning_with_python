# -*- coding: utf-8 -*-
"""
Deep Learning with Python by Francois Chollet
3. Getting started with neural networks
3.4 Classifying movie reviews: a binary classification example
"""
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

# Loading the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Encoding the integer sequences into a binary matrix
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# The model definition
model = Sequential()
model.add(Dense(16, activation='relu'), input_shape=(10000,))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))