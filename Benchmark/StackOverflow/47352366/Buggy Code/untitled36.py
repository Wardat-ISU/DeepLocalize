#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:37:13 2020

@author: wardat
"""

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import keras
from time import *
import sys



import keras
from keras.models import Sequential

model = Sequential()

bias_initializer = keras.initializers.Constant(value = 0.1)

neurons_nb_layer_1 = 32
neurons_nb_layer_2 = 64
neurons_nb_layer_3 = 1024

start_time = time()

from keras.layers import Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
model.add(Reshape((28, 28, 1), input_shape=(784,)))
model.add(Conv2D(filters = neurons_nb_layer_1, kernel_size =(5,5), padding = 'same', activation = "relu", bias_initializer = bias_initializer))
model.add(MaxPooling2D(padding='same'))
model.add(Conv2D(filters = neurons_nb_layer_2, kernel_size = (5,5), padding = 'same', activation = "relu", bias_initializer = bias_initializer))
model.add(MaxPooling2D(padding='same'))
model.add(Reshape((1,7*7*neurons_nb_layer_2)))
model.add(Dense(units = neurons_nb_layer_3, activation = "relu", bias_initializer = bias_initializer))
model.add(Dropout(rate = 0.5))
model.add(Flatten())
model.add(Dense(units = 10, activation = "relu"))

#model.summary()

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics=['accuracy']
              )


import datetime
start2 = datetime.datetime.now()
for i in range(600):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = model.evaluate(batch[0], batch[1])
        print("step", i, ":", train_accuracy)
    model.train_on_batch(batch[0], batch[1])

end_time = time()

end2 = datetime.datetime.now()
time2 = (end2 - start2).seconds
print(time2//60, "min", time2%60,"s")

print("time",(end_time - start_time))
sys.exit(1)