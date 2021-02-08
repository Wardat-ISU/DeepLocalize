#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:42:21 2020

@author: wardat
"""
# Multiclass Classification with the Iris Flowers Dataset 
import numpy 
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense , Activation,Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.utils import np_utils 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold 
from sklearn.preprocessing import LabelEncoder 
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD
import pandas
import numpy as np

# seed weights
np.random.seed(3)

dataframe = pd.read_csv('dataset.csv', delimiter=',')
data = dataframe.values 
X_train = data[:,0:16].astype(float) 
Y = data[:,16]

# encode class values as integers 
encoder = LabelEncoder() 
encoder.fit(Y) 
encoded_Y = encoder.transform(Y) 

# convert integers to dummy variables (i.e. one hot encoded) 
y_train = np_utils.to_categorical(encoded_Y) 

print(X_train)
model = Sequential()
model.add(BatchNormalization())
model.add(Dense(64, input_dim=16, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, init='uniform'))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy' ])
model.fit(X_train, y_train, nb_epoch=20, batch_size=16)

model.evaluate(X_train, y_train)