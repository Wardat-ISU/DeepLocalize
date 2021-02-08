from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy
import keras
import time

X = numpy.array([[1., 1.], [0., 0.], [1., 0.], [0., 1.], [1., 1.], [0., 0.]])
y = numpy.array([[0.], [0.], [1.], [1.], [0.], [0.]])
model = Sequential()
model.add(Dense(2, input_dim=2, init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(3, init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(1, init='uniform'))
model.add(Activation('softmax'))
start_time =time.clock()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(X, y, nb_epoch=20 ,     
        callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(X, y, len(model.layers), batch_size =32,  startTime =start_time )
]
  )
end_time = time.clock()
print("time per second",(end_time-start_time) )
score = model.evaluate(X, y)
print()
print(score)
print(model.predict(numpy.array([[1, 0]])))
print(model.predict(numpy.array([[0, 0]])))