# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 04:56:35 2020

@author: PC_Wardat
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from time import *
from keras import backend as K
import sys
import keras

training_data =np.array([1,2,3,4])

result_data =np.array([2,4,6,8])


training_data = training_data.reshape(4,1)
result_data = result_data.reshape(4,1)
model = Sequential([
    Dense(1, activation="linear", input_shape=(1,))
])

model.compile(loss="mean_squared_error", optimizer="SGD")
model.fit(training_data, result_data, epochs=20, verbose=0)

outputs = []
for layer in model.layers:
    keras_function = K.function([model.layers[0].input], [layer.output])
    outputs.append(keras_function([training_data, 1]))
print(outputs)