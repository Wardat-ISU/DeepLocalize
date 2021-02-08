#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:34:23 2020

@author: wardat
"""
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import make_multilabel_classification
from keras import Sequential
from keras.layers import Dense, Flatten, Masking, LSTM, GRU, Conv1D, Dropout, MaxPooling1D
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from time import *
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

start_time = time()

layers = [10,20,30,40,50]
model = keras.models.Sequential()
#Stacking Layers
model.add(keras.layers.Dense(layers[0], input_dim = m, activation='relu'))
#Defining the shape of input
for layer in layers[1:]:
    model.add(keras.layers.Dense(layer, activation='relu'))
    #Layer activation function
# Output layer
model.add(keras.layers.Dense(2, activation='softmax'))
#Pre-training
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#Training
model.fit(x_train, y_train, validation_split = 0.10, epochs = 50, batch_size = 10, shuffle = True, verbose = 2)
end_time = time()
# evaluate the network
loss, accuracy = model.evaluate(x_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
#predictions
predt = model.predict(x_test)
print(predt)

print("time", (end_time - start_time))
sys.exit(1)