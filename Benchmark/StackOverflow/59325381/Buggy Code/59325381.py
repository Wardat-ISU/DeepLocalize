import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Explore data
print(y_train[12])
print(np.shape(x_train))
print(np.shape(x_test))
#we have 60000 imae for the training and 10000 for testing

# Scaling data
x_train = x_train/255
y_train = y_train/255
batch_size = 1
#reshape the data
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)

start_time = time.clock()
#Create a model
model = keras.Sequential([
keras.layers.Conv2D(64,(3,3),strides= (1,1),padding = "same",input_shape=(28,28,1)),
keras.layers.MaxPooling2D(pool_size = (2,2),padding = "valid"),
keras.layers.Conv2D(32,(3,3),strides= (1,1),padding = "same"),
keras.layers.MaxPooling2D(pool_size = (2,2),padding = "valid"),
keras.layers.Flatten(),
keras.layers.Dense(128),
keras.layers.Activation("relu"),
keras.layers.Dense(10),
keras.layers.Activation("softmax")])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics  = ['accuracy'])

model.fit(x_train,y_train,epochs=2 ,
          callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(x_train,y_train, len(model.layers), batch_size, start_time )  
]
)

test_loss,test_acc = model.evaluate(x_test,y_test)
print("\ntest accuracy:",test_acc)
end_time = time.clock()

print("time", (end_time - start_time))
sys.exit(1)