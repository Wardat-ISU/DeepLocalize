#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:34:23 2020

@author: wardat
"""
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import make_multilabel_classification
from keras import Sequential
from keras.layers import Dense, Flatten, Masking, LSTM, GRU, Conv1D, Dropout, MaxPooling1D, Activation
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import time 
import keras
import sys


n = 1000         # Number of instances
m = 4            # Number of features
num_classes = 1  # Number of output classes

... # Your code for loading the data
dummyX, dummyY = make_multilabel_classification(n_samples=n, n_features=m, n_classes=num_classes)
labelEncoder = LabelEncoder()
dummyY = labelEncoder.fit_transform(dummyY)

x_train, x_test, y_train, y_test = train_test_split(dummyX, dummyY, test_size=0.20)


input_shape = (m,)

batch_size = 10 
start_time = time.clock()
layers = [10,20,30,40,50]
model = keras.models.Sequential()
#Stacking Layers
model.add(keras.layers.Dense(layers[0], input_dim = m, activation='relu'))
#Defining the shape of input
for layer in layers[1:]:
    model.add(keras.layers.Dense(layer))
    model.add(Activation('relu'))
    #Layer activation function
# Output layer
model.add(keras.layers.Dense(1))
model.add(Activation('sigmoid'))
#Pre-training
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#Training
model.fit(x_train, y_train, validation_split = 0.10, epochs = 50, batch_size = 10, shuffle = True, verbose = 1,
callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(x_train, y_train, len(model.layers),batch_size, start_time)
]
          
)
end_time = time.clock()
print("time", (end_time - start_time))
sys.exit(1)