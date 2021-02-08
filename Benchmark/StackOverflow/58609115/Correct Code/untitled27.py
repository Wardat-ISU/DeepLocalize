#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:09:24 2020

@author: wardat
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import time 
import sys
import keras

df = pd.read_csv("export78.csv")

onehotencoder = OneHotEncoder(categorical_features = [1])
data2 = onehotencoder.fit_transform(df).toarray()
dataset = pd.DataFrame(data2)

#print(dataset)
X= dataset.iloc[:,69].astype(float)
y= dataset.iloc[:,0:69].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("shape")
print(X_train.shape)
start_time = time.clock()
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(69, kernel_initializer='random_normal', input_dim=1))
#Second  Hidden Layer
classifier.add(Activation('relu'))
classifier.add(Dense(69, kernel_initializer='random_normal'))
#Output Layer
classifier.add(Activation('relu'))
classifier.add(Dense(69, kernel_initializer='random_normal'))
classifier.add(Activation('softmax'))
#Compiling the neural network
classifier.compile(optimizer ='adam',loss='categorical_crossentropy', metrics=['accuracy'])
X = X_train.values. reshape(1767,1)

batch_size = 50
#X_train =np.reshape( X_train, (1767,1))
#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=50, epochs=10,
callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.backpropagation(X,y_train, len(classifier.layers),batch_size, start_time)
]
        )

end_time = time.clock()
accr = classifier.evaluate(X_test, y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0] ,accr[1]))


classifier.save("model.h67")
print("time =",(end_time - start_time))

data1 = np.array(X_test)
List = [data1]
model = tf.keras.models.load_model("model.h67")
prediction = model.predict([(data1)])
target = (np.argmax(prediction, axis=0))
dataset1 = pd.DataFrame(target)

print(dataset1)

sys.exit(1)