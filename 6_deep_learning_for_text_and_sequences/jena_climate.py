# -*- coding: utf-8 -*-
"""
Deep Learning with Python by Francois Chollet
6. Deep learning for text and sequences
6.3 Advanced use of recurrent neural networks
6.3.1 A temperature-forecasting problem
"""
# Inspecting the data of the Jena weather dataset
import os

data_dir = 'jena_climate'
fname = os.path.join(data_dir, "jena_climate_2009_2016.csv")

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))


# Parsing the data
import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values


# Plotting the temperature timeseries
import matplotlib.pyplot as plt

temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)


# Plotting the first 10 days of the temperature timeseries
plt.figure()
plt.plot(range(1440), temp[:1440])


# Normalizing the data
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


# Generator yielding timeseries samples and their targets
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index,
                                     size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((batch_size, lookback//step, data.shape[-1]))
        targets = np.zeros((batch_size,))
        
        for j, row in enumerate(rows):
            indices = range(row - lookback, row, step)
            samples[j] = data[indices]
            targets[j] = data[row + delay][1]
        
        return samples, targets


# Preparing the training, validation and test generators
lookback = 1440
step = 6
delay = 144
batch_size = 128


