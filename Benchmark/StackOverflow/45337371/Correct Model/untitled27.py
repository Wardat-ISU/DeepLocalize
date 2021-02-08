#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:12:12 2020

@author: wardat
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import numpy as np
num_features = 5
import tensorflow as tf
from tensorflow import keras

model = Sequential()
model.add(Dense(60, input_shape=(num_features,), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')
#Run predict to initialize weights
model.predict(np.random.rand(1, num_features))

x = tf.random_uniform(shape=(1, num_features))
model_grad = tf.gradients(model(x), x)[0]
sess = K.get_session()
print( model_grad.eval(session=sess) )
