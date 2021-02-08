import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import time 
from keras import backend as K
import sys
import keras


df = pd.read_csv("export78.csv")

onehotencoder = OneHotEncoder(categorical_features = [1])
data2 = onehotencoder.fit_transform(df).toarray()
dataset = pd.DataFrame(data2)

X= dataset.iloc[:,69].astype(float)
y= dataset.iloc[:,0:69].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

batch_size = 50
print(X_train.shape)
start_time = time.clock()
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(69, kernel_initializer='random_normal', input_dim=1))
#Second  Hidden Layer
classifier.add(Activation('sigmoid'))
classifier.add(Dense(69, kernel_initializer='random_normal'))
#Output Layer
classifier.add(Activation('sigmoid'))
classifier.add(Dense(69, kernel_initializer='random_normal'))
classifier.add(Activation('sigmoid'))
#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics=['accuracy'])


X = X_train.values. reshape(1767,1)

#Fitting the data to the training dataset
classifier.fit(X_train,  y_train, batch_size=50, epochs=1,
callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(X,  y_train, len(classifier.layers), batch_size,  start_time)
]
               )
end_time = time.clock()

from keras import backend as K
import numpy as np

from keras import backend as K


outputs = []
for layer in classifier.layers:
    keras_function = K.function([classifier.layers[0].input], [layer.output])
    outputs.append(keras_function([X, 1]))
print(outputs)
