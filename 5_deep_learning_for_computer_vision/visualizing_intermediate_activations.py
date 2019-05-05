# -*- coding: utf-8 -*-
"""
Deep Learning with Python by Francois Chollet
5. Deep learning for computer vision
5.4 Visualizing what convnets learn
5.4.1 Visualizing intermediate activations
"""
from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')
model.summary()


# Preprocessing a single image
img_path = 'cats_and_dogs_small/test/cats/cat.1700.jpg'

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.


# Displaying the test picture
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()


# Instantiating a model from an input tensor and a list of output tensors
from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)


# Running the model in predict mode
activations = activation_model.predict(img_tensor)


# Visualizing the 4th channel
import matplotlib.pyplot as plt
plt.matshow(activations[0][0, :, :, 4], cmap='viridis')

# Visualizing the 7th channel
import matplotlib.pyplot as plt
plt.matshow(activations[0][0, :, :, 7], cmap='viridis')


# Visualizing every channel in every intermediate activation
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    
    size = layer_activation.shape[1]
    
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size*n_cols, images_per_row*size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col*images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col*size : (col + 1)*size,
                         row*size : (row + 1)*size] = channel_image
    
    scale = 1./size
    plt.figure(figsize=(scale*display_grid.shape[1],
                        scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


