# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:56:55 2019

@author: PC_Wardat
"""
import time
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.initializers import RandomNormal

# import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# input image dimensions
img_rows, img_cols = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
input_shape = (img_rows * img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
num_classes = 10
batch_size = 2000
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)

model = Sequential()
model.add(Dense(5000, input_dim=x_train.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(600))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
start_time = time.time()

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy',])

model.fit(x_train, y_train,epochs=5,batch_size=2000,
          callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
keras.callbacks.DeepLocalize(x_train, y_train, len(model.layers),batch_size)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
]
)
end_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))
score = model.evaluate(x_test, y_test, batch_size=2000)
print(score)

