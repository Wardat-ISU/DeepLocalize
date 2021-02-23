from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.datasets import mnist

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
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(dummyX, dummyY, test_size=0.20)

model.fit(X_train, y_train,nb_epoch=20, batch_size=20, validation_data=(X_test, y_test))



