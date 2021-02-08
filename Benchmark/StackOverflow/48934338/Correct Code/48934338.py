#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:34:43 2020

@author: wardat
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import keras
from matplotlib import pyplot as plt
from keras import optimizers

# fix random seed for reproducibility
seed = 7
#datapoints
X = np.arange(0.0, 5.0, 0.1, dtype='float32').reshape(-1,1)
y = 5 * np.power(X,2) + np.power(np.random.randn(50).reshape(-1,1),3)

#model
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=1))
model.add(Dense(30, activation='relu', init='uniform'))
model.add(Dense(output_dim=1, activation='linear'))

#training
sgd = optimizers.SGD(lr=0.001)
model.compile(loss='mse', optimizer=sgd)
model.fit(X, y, nb_epoch=1000)

#predictions
predictions = model.predict(X)

#plot
plt.scatter(X, y,edgecolors='g')
plt.plot(X, predictions,'r')
plt.legend([ 'Predictated Y' ,'Actual Y'])
plt.show()