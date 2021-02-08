#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:15:55 2020

@author: wardat
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
from math import ceil
import keras
seed = 7
random.seed(seed)
#Dimension of layers
dim = 8

#Generate dataset
X = []
for i in range(0,2**dim):
    n = [float(x) for x in bin(i)[2:]]
    X.append([0.]*(dim-len(n))+n)
y = X[:]
#random.shuffle(y)
X = np.array(X)
y = np.array(y)

# create model
model = Sequential()
model.add(Dense(dim, input_dim=dim, init='normal', activation='relu'))
model.add(Dense(dim, init='normal', activation='relu'))
model.add(Dense(dim, init='normal', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
# Fit the model
model.fit(X, y, nb_epoch=1000, batch_size=1, verbose=1)
# evaluate the model
scores = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
output = model.predict(X)

#Make the output binary
for i in range(0, output[:,0].size):
    for j in range(0, output[0].size):
        if output[i][j] > 0.5 or output[i][j] == 0.5:
            output[i][j] = 1
        else:
            output[i][j] = 0
print(output)