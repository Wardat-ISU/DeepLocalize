#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:50:34 2020

@author: wardat
"""

from keras.models import Sequential
from keras.layers import Dense,BatchNormalization
from sklearn import preprocessing
import numpy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
seed = 4
numpy.random.seed(seed)

dataset = numpy.loadtxt("NetworkPackets.csv", delimiter=",")
X = dataset[:, 0:11].astype(float)
Y = dataset[:, 11]
X = preprocessing.scale(X)

model = Sequential()
model.add(Dense(12, input_dim=11, kernel_initializer='normal', activation='relu'))
model.add(Dense(12, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X, Y, nb_epoch=100, batch_size=5)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))