# -*- coding: utf-8 -*-
"""
Deep Learning with Python by Francois Chollet
5. Deep learning for computer vision
5.3 Using a pretrained convnet
Feature extraction with data augmentation
"""
# Instantiating the VGG16 convolutional base
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

conv_base.summary()