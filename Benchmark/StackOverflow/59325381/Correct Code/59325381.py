import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Explore data
print(y_train[12])
print(np.shape(x_train))
print(np.shape(x_test))
#we have 60000 imae for the training and 10000 for testing

# Scaling data
x_train = x_train/255
x_test = x_test/255
#reshape the data
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
#Create a model
model = keras.Sequential([
keras.layers.Conv2D(64,(3,3),(1,1),padding = "same",input_shape=(28,28,1)),
keras.layers.MaxPooling2D(pool_size = (2,2),padding = "valid"),
keras.layers.Conv2D(32,(3,3),(1,1),padding = "same"),
keras.layers.MaxPooling2D(pool_size = (2,2),padding = "valid"),
keras.layers.Flatten(),
keras.layers.Dense(128,activation = "relu"),
keras.layers.Dense(10,activation = "softmax")])

model.compile(optimizer = "adam",
loss = "categorical_crossentropy",
metrics  = ['accuracy'])

model.fit(x_train,y_train,epochs=10)
test_loss,test_acc = model.evaluate(x_test,y_test)
print("\ntest accuracy:",test_acc)