# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:43:51 2019

@author: PC_Wardat
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
from keras.initializers import RandomNormal
import tensorflow as tf
import numpy
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
print (x_tr.shape)

X_train = numpy.array([[1] * 128] * (10 ** 4) + [[0] * 128] * (10 ** 4))
X_test = numpy.array([[1] * 128] * (10 ** 2) + [[0] * 128] * (10 ** 2))

Y_train = numpy.array([True] * (10 ** 4) + [False] * (10 ** 4))
Y_test = numpy.array([True] * (10 ** 2) + [False] * (10 ** 2))
print(Y_train.shape)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

Y_train = Y_train.astype("bool")
Y_test = Y_test.astype("bool")
batch_size =1
nb_epoch =3

model = Sequential()
model.add(Dense(units=50,activation='relu', input_dim=128))
model.add(Dropout(0.2))
model.add(Dense(units=50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
rms = tf.keras.optimizers.RMSprop()
model.compile(loss='binary_crossentropy', optimizer=rms,metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, epochs=3, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


