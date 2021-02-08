from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import random
from math import ceil
import time
import sys
import keras

#Dimension of layers
dim = 8

#Generate dataset
X = []
for i in range(0,2**dim):
    n = [float(x) for x in bin(i)[2:]]
    X.append([0.]*(dim-len(n))+n)
y = X[:]
random.shuffle(y)
X = np.array(X)
y = np.array(y)

# create model
model = Sequential()
model.add(Dense(dim, input_dim=dim, init='normal'))
model.add(Activation('sigmoid'))
model.add(Dense(dim, init='normal'))
model.add(Activation('sigmoid'))
model.add(Dense(dim, init='normal'))
model.add(Activation('sigmoid'))

start_time = time.clock()
# Compile model
model.compile(loss='mse', optimizer='SGD', metrics=['accuracy'])
# Fit the model
model.fit(X, y, nb_epoch=1000, batch_size=50, verbose=1,
          callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(X, y, len(model.layers), batch_size=50, startTime = start_time)
]
          )
# evaluate the model
scores = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
output = model.predict(X)
end_time = time.clock()


#Make the output binary
for i in range(0, output[:,0].size):
    for j in range(0, output[0].size):
        if output[i][j] > 0.5 or output[i][j] == 0.5:
            output[i][j] = 1
        else:
            output[i][j] = 0
print(output)
print("Time", (end_time -start_time))
sys.exit(1)