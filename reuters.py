# -*- coding: utf-8 -*-
"""
Deep Learning with Python by Francois Chollet
3. Getting started with neural networks
3.5 Classifying newswires: a multiclass classification example
"""
from keras.datasets import reuters
from keras.utils import to_categorical
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

# Loading the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Encoding the data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

