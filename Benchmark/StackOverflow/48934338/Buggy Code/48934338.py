#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:34:43 2020

@author: wardat
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras
from matplotlib import pyplot as plt
from keras import optimizers
import time 
import sys

# fix random seed for reproducibility
seed = 7
#datapoints
X = np.arange(0.0, 5.0, 0.1, dtype='float32').reshape(-1,1)
y = 5 * np.power(X,2) + np.power(np.random.randn(50).reshape(-1,1),3)

#model
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Activation('relu'))
model.add(Dense(30, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(output_dim=1))
model.add(Activation('linear'))

start_time = time.clock()
#training
sgd = optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
model.fit(X, y, nb_epoch=1000,
          callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(X, y, len(model.layers), batch_size =32, startTime = start_time )
]
)

end_time = time.clock()


#predictions
predictions = model.predict(X)

print("time estimated",(end_time -start_time))
#plot
plt.scatter(X, y,edgecolors='g')
plt.plot(X, predictions,'r')
plt.legend([ 'Predictated Y' ,'Actual Y'])
plt.show()
sys.exit(1)