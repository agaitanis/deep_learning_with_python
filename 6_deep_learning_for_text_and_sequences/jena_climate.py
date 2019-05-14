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