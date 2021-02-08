#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:31:42 2020

@author: wardat
"""

import numpy 
import pandas 
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.utils import np_utils 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold 
from sklearn.preprocessing import LabelEncoder 
from sklearn.pipeline import Pipeline 
import time
from sklearn.preprocessing import MinMaxScaler
import keras
import sys
# fix random seed for reproducibility 
seed = 7 
numpy.random.seed(seed) 
# load dataset 
dataframe = pandas.read_csv("iris.data", header=None) 
dataset = dataframe.values 
X = dataset[:,0:4].astype(float) 
Y = dataset[:,4] 

# encode class values as integers 
encoder = LabelEncoder() 
encoder.fit(Y) 
encoded_Y = encoder.transform(Y) 

# convert integers to dummy variables (i.e. one hot encoded) 
dummy_y = np_utils.to_categorical(encoded_Y) 
batch_size = 5
start_time = time.clock()
# define baseline model 


# create model
model = Sequential()
model.add(Dense(4, input_dim=4, kernel_initializer="normal"))
model.add(Activation('relu'))
model.add(Dense(3, kernel_initializer="normal"))
model.add(Activation('sigmoid'))

# Compile model
model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])


model.fit( X, dummy_y, nb_epoch=200, batch_size=5, verbose=1,
callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(X, dummy_y, len(model.layers), batch_size, start_time)
]    ) 



end_time = time.clock()

print("time", (end_time - start_time))
sys.exit(1)