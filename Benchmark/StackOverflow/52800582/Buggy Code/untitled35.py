#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:49:37 2020

@author: wardat
"""

import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
import math
import time
import sys
import keras


x = np.arange(-100, 100, 0.5)
y = x**4

x_train = x.reshape(400,1)


model = Sequential()
model.add(Dense(50, input_shape=(1,)))
model.add(Activation('sigmoid'))
model.add(Dense(50) )
model.add(Activation('elu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

t1 = time.clock()
for i in range(1):
    model.fit(x, y, epochs=1000, batch_size=len(x), verbose=1,
              callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(x_train, y, len(model.layers), batch_size=len(x), startTime = t1 )              
]
)
    predictions = model.predict(x)
    print (i," ", np.mean(np.square(predictions - y))," t: ", time.clock()-t1)

    #plt.hold(False)
    plt.plot(x, y, 'b', x, predictions, 'r--')
    #plt.hold(True)
    plt.ylabel('Y / Predicted Value')
    plt.xlabel('X Value')
    plt.title([str(i)," Loss: ",np.mean(np.square(predictions - y))," t: ", str(time.clock()-t1)])
    plt.pause(0.001)

#plt.savefig("fig2.png")
plt.show()
t2 = time.clock()
print("time",(t2-t1))
sys.exit(1)