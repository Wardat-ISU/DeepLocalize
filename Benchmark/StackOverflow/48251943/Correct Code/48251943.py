from keras.models import Sequential
from keras.layers import Dense
import numpy
import matplotlib.pyplot as plt

#fix random seed for reproducibility
numpy.random.seed(7)

#load and read dataset
dataset = numpy.genfromtxt("Book1.csv", delimiter=',')
# split into input (X) and output (Y) variables
X = dataset[:,2:4]
print(X.shape)
Y = dataset[:,1]
print ("Variables: \n", X)
print ("Target_outputs: \n", Y)
# create model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
#model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='tanh'))
model.summary()
# Compile model
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['MSE'])
# Fit the model
history  = model.fit(X, Y, epochs=500, batch_size=10)
#make predictions (test)
F = model.predict(X)
print ("Predicted values: \n", F)
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()