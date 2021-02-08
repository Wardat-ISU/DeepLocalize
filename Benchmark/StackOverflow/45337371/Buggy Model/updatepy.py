# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 07:09:23 2020

@author: PC_Wardat
"""

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from time import *
import time 
import numpy as np
num_features = 5
batch_size = 1
epochs = 5
x = np.random.uniform(size=( 1000, num_features, 1))
y = np.random.uniform(size=( 1000, num_features,1))
model = Sequential()
model.add(Dense(60, input_shape=(num_features,1)))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))

start_time = time.clock()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x, y,batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(x, y, len(model.layers), batch_size=batch_size, startTime = start_time  )
])

end_time = time.clock()
print("Run time", (end_time - start_time))