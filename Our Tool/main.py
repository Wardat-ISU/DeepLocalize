import numpy as np
from tensorflow.python.keras.datasets import mnist

from model.loss import *
from model.layers import *
from model.network import *

print('Loadind data......')
num_classes = 10
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('Preparing data......')
#train_images -= int(np.mean(train_images))
#train_images = train_images /int(np.std(train_images))
#test_images -= int(np.mean(test_images))
#test_images = test_images / int(np.std(test_images))
train_images = train_images / 255
test_images  = test_images / 255
training_data = train_images.reshape(60000, 1, 28, 28)
training_labels = np.eye(num_classes)[train_labels]
testing_data = test_images.reshape(10000, 1, 28, 28)
testing_labels = np.eye(num_classes)[test_labels]
lr = 0.01
checker = False
net = Sequential()
net.add(Flatten())
net.add(Dense(num_inputs=784, num_outputs=84, learning_rate=lr, name='fc6'))
net.add(ReLu())
net.add(Dense(num_inputs=84, num_outputs=10, learning_rate=lr, name='fc7'))
net.add(Softmax())
print('Training Lenet......')
net.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
net.fit(training_data, training_labels, 1, 1)

# learn rate should be 0.01