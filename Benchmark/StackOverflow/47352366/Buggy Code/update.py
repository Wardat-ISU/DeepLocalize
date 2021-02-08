# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 05:34:11 2020

@author: PC_Wardat
"""

from keras.datasets import mnist
import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()

print(x_tr.shape)
import keras
import time
import sys
import keras
from keras.models import Sequential
from keras.layers import Activation
model = Sequential()
bias_initializer = keras.initializers.Constant(value = 0.1)

neurons_nb_layer_1 = 32
neurons_nb_layer_2 = 64
neurons_nb_layer_3 = 1024
img_rows, img_cols = 28, 28
num_classes = 10 
batch_size = 50

start_time = time.clock()
y_train = keras.utils.to_categorical(y_tr, num_classes)

print(y_tr.shape)
x_train = x_tr.reshape(x_tr.shape[0], img_rows* img_cols)
x_test = x_te.reshape(x_te.shape[0], img_rows* img_cols)
from keras.layers import Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
model.add(Reshape((28, 28, 1), input_shape=(784,)))
model.add(Conv2D(filters = neurons_nb_layer_1, kernel_size =(5,5), padding = 'same', bias_initializer = bias_initializer))
model.add(Activation ( "relu"))
model.add(MaxPooling2D(padding='same'))
model.add(Conv2D(filters = neurons_nb_layer_2, kernel_size = (5,5), padding = 'same', bias_initializer = bias_initializer))
model.add(Activation ( "relu"))
model.add(MaxPooling2D(padding='same'))
model.add(Reshape((1,7*7*neurons_nb_layer_2)))
model.add(Dense(units = neurons_nb_layer_3, bias_initializer = bias_initializer))
model.add(Activation ( "relu"))
model.add(Dropout(rate = 0.5))
model.add(Flatten())
model.add(Dense(units = 10))
model.add(Activation ( "relu"))

#model.summary()

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics=['accuracy']
              )

model.fit(x_train, y_train, batch_size=50, epochs=5, verbose=1,
          callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(x_train, y_train, len(model.layers), batch_size, start_time )  
])

end_time = time.clock()
print("time",(end_time - start_time))
sys.exit(1)
