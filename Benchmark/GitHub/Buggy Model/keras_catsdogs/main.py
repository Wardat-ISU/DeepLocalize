#!/usr/bin/env python3
#coding=UTF-8

# modified from https://www.youtube.com/watch?v=cAICT4Al5Ow

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD
from keras import backend as K
from scipy.misc import imread, imsave, imresize
import numpy as np
import glob
import sys
import os.path

img_width = 32
img_heigth = 32


filelist = glob.glob('/Users/ronny/Downloads/__new/Stanford Dogs Dataset/example32/*')

dogs = np.array([np.array(imread(filename).flatten()) for filename in filelist])
dogs_y = np.zeros((len(filelist),1))
filelist = glob.glob('/Users/ronny/Downloads/__new/Cats Dataset/example32/*')
cats = np.array([np.array(imread(filename).flatten()) for filename in filelist])
cats_y = np.ones((len(filelist),1))

both = np.concatenate((dogs, cats), axis=0)
both_y = np.concatenate((dogs_y, cats_y), axis=0)

np.random.seed(1)
np.random.shuffle(both)
np.random.seed(1)
np.random.shuffle(both_y)

print("done loading data")

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=3072))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print("done compiling the model")

model.fit(both, both_y, epochs=10, verbose=1)

#model.save_weights ...

print("done with fitting")
