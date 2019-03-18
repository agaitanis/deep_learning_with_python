# -*- coding: utf-8 -*-
"""
Deep Learning with Python by Francois Chollet
3. Getting started with neural networks
3.6 Predicting house prices: a regression example
"""
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


# Loading the Boston housing dataset
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalizing the data
m = np.mean(train_data, axis=0)
s = np.std(train_data, axis=0)
train_data = (train_data - m)/s
test_data = (test_data - m)/s

# Model definition
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='mse',           # Mean Square Error
                  metrics=['mae'])      # Mean Absolute Error
    return model

# K-fold validation
k = 4
