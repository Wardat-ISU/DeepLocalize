from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.datasets import mnist
import time
import sys
# generate some data
dummyX, dummyY = make_multilabel_classification(n_samples=60000, n_features=20, n_classes=3)

# neural network
model = Sequential()
model.add(Dense(20, input_dim=20))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error',
          optimizer='sgd',
          metrics=['accuracy'])

start_time = time.clock()
X_train, X_test, y_train, y_test = train_test_split(dummyX, dummyY, test_size=0.20)

model.fit(X_train, y_train,nb_epoch=20, batch_size=20, validation_data=(X_test, y_test),
          callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(X_train, y_train, len(model.layers),batch_size=20, startTime = start_time )
]
)

end_time = time.clock()
print("time",(end_time -start_time))

sys.exit(1)

