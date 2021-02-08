#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:20:03 2020

@author: wardat
"""
from keras.layers import Dense
from keras.models import Sequential
import keras.utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
import keras



df = pd.read_csv("adult.data",header=None)
X = df.iloc[:,0:14]
Y = df.iloc[:,14]

encoder = LabelEncoder()
#X
for i in [1,3,5,6,7,8,9,13]:
   column = X[i]
   encoder.fit(column)
   encoded_C = encoder.transform(column)
   X[i] = keras.utils.to_categorical(encoded_C)

print(X.shape)
#Y
encoder.fit(Y)
en_Y = encoder.transform(Y)
Y = keras.utils.to_categorical(en_Y)

#model
model = Sequential()
model.add(Dense(21, input_dim=14, activation="relu"))
model.add(Dense(2, activation="softmax"))
#compile
model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

#train
model.fit(X,Y, epochs=50, batch_size=100)
score = model.evaluate(X,Y)
print("Accuracy: {}%".format(score[0]))