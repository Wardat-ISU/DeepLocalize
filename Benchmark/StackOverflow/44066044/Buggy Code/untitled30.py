#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:50:34 2020

@author: wardat
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import preprocessing
import numpy
import os
import time 
import sys
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
seed = 4
numpy.random.seed(seed)

dataset = numpy.loadtxt("NetworkPackets.csv", delimiter=",")
X = dataset[:, 0:11].astype(float)
Y = dataset[:, 11]
batch_size = 5
print(X.shape)
print(Y.shape)
start_time = time.clock()
model = Sequential()
model.add(Dense(12, input_dim=11, kernel_initializer='normal'))
model.add(Activation('relu'))
model.add(Dense(12, kernel_initializer='normal'))
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.add(Activation('relu'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X, Y, nb_epoch=100, batch_size=5,
callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(X, Y, len(model.layers),batch_size, start_time )
]
)
end_time = time.clock()
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("time",(end_time - start_time))
sys.exit(1)