
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split

import keras
import time

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# Import the dataset
dataset = pd.read_csv("Linear_Data.csv", header=None).values
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:1], dataset[:,1], 
                                                    test_size=0.25)
start_time = time.clock()
# Now we build the model
neural_network = Sequential() # create model
neural_network.add(Dense(5, input_dim=1)) # hidden layer
neural_network.add(Activation('sigmoid'))
neural_network.add(Dense(1)) # output layer
neural_network.add(Activation('sigmoid'))
neural_network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
neural_network_fitted = neural_network.fit(X_train, Y_train, epochs=1000, verbose=1, 
                                           batch_size=X_train.shape[0], initial_epoch=0,
   callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(X_train, Y_train, len(neural_network.layers), batch_size=X_train.shape[0],startTime =start_time )
]
)
end_time = time.clock()
from keras import backend as K

print("time",(end_time - start_time))
outputs = []
for layer in neural_network.layers:
    keras_function = K.function([neural_network.layers[0].input], [layer.output])
    outputs.append(keras_function([X_train, 1]))
print(outputs)