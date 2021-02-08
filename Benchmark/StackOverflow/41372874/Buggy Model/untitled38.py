import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import time
import sys
import keras
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset
dataframe = pandas.read_csv('SIN.csv', usecols=[0,1,2], engine='python')
dataset = dataframe.values

X = dataset[:,0:2].astype(float)
Y = dataset[:,2].astype(int)

# split into train and test sets
train_size = int(len(X) * 0.80)
test_size = len(X) - train_size
Xtrain, Xtest = X[0:train_size,:], X[train_size:len(X),:]
Ytrain, Ytest = Y[0:train_size], Y[train_size:len(Y)]


print(len(Xtrain), len(Xtest))

start_time= time.clock()
# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(2, input_dim=2, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(2, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, Ytrain, nb_epoch=20, batch_size=2, verbose=1,
callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(Xtrain, Ytrain, len(model.layers), batch_size=2, startTime = start_time )
]
  )
end_time= time.clock()
# Estimate model performance
trainScore = model.evaluate(Xtrain, Ytrain, verbose=2)
print('Train Score: %.2f' % trainScore[1])
testScore = model.evaluate(Xtest, Ytest, verbose=2)
print('Test Score: %.2f' % testScore[1])
print("time", (end_time -start_time))
sys.exit(1)