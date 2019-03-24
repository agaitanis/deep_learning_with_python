# -*- coding: utf-8 -*-
"""
Deep Learning with Python by Francois Chollet
4. Fundamentals of machine learning
4.4 Overfitting and underfitting
"""
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# Original model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

original_loss_values = history.history['loss']
original_val_loss_values = history.history['val_loss']
epochs = range(1, len(original_val_loss_values) + 1)


# Version of the model with lower capacity
model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

val_loss_values = history.history['val_loss']

fig = plt.figure()
ax = fig.gca()
ax.plot(epochs, original_val_loss_values, '+', label='Original model')
ax.plot(epochs, val_loss_values, 'o', label='Smaller model')
ax.set_xlabel('Epochs')
ax.set_ylabel('Validation Loss')
ax.legend()


# Version of the model with higher capacity
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

loss_values = history.history['loss']
val_loss_values = history.history['val_loss']

fig = plt.figure()
ax = fig.gca()
ax.plot(epochs, original_val_loss_values, '+', label='Original model')
ax.plot(epochs, val_loss_values, 'o', label='Bigger model')
ax.set_xlabel('Epochs')
ax.set_ylabel('Validation Loss')
ax.legend()

fig = plt.figure()
ax = fig.gca()
ax.plot(epochs, original_loss_values, '+', label='Original model')
ax.plot(epochs, loss_values, 'o', label='Bigger model')
ax.set_xlabel('Epochs')
ax.set_ylabel('Training Loss')
ax.legend()


# Adding L2 weight regularization to the model
model = Sequential()
model.add(Dense(16, kernel_regularizer=l2(0.001), activation='relu', 
                input_shape=(x_train.shape[1],)))
model.add(Dense(16, kernel_regularizer=l2(0.001), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

val_loss_values = history.history['val_loss']

fig = plt.figure()
ax = fig.gca()
ax.plot(epochs, original_val_loss_values, '+', label='Original model')
ax.plot(epochs, val_loss_values, 'o', label='L2-regularized model')
ax.set_xlabel('Epochs')
ax.set_ylabel('Validation Loss')
ax.legend()


# Adding dopout to the IMDB network
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

val_loss_values = history.history['val_loss']

fig = plt.figure()
ax = fig.gca()
ax.plot(epochs, original_val_loss_values, '+', label='Original model')
ax.plot(epochs, val_loss_values, 'o', label='Dropout-regularized model')
ax.set_xlabel('Epochs')
ax.set_ylabel('Validation Loss')
ax.legend()
