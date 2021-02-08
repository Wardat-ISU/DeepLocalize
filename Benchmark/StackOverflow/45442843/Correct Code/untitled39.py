#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:01:07 2020

@author: wardat
"""
import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from sklearn import preprocessing
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset
dataframe = pandas.read_csv('dataset.csv', delimiter=",")
dataset = dataframe.values


X = dataset[:,0:1].astype(float)
Y = dataset[:,1].astype(bool)


X = preprocessing.scale(X)
# split into train and test sets
train_size = int(len(X) * 0.80)
test_size = len(X) - train_size
Xtrain, Xtest = X[0:train_size,:], X[train_size:len(X),:]
Ytrain, Ytest = Y[0:train_size], Y[train_size:len(Y)]


print(Xtrain.shape)
print(len(Xtrain), len(Xtest))


model = Sequential()
model.add(Dense(1 , input_dim =1 ))
model.add(Activation('sigmoid'))
model.add(Dense(1 , init='normal' , activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest),epochs=100,batch_size=200,verbose=2)
model.evaluate(Xtest,Ytest)