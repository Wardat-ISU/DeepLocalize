from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy
import time
import keras

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

start_time = time.clock()
# create model
model = Sequential()
model.add(Dense(12, input_dim=8))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
#model.add(Dense(120, activation='relu'))
#model.add(Dense(36, activation='relu'))
#model.add(Dense(12, activation='relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
# Fit the model

model.fit(X, Y, epochs=150, batch_size=10,
          callbacks= [
#keras.callbacks.TerminateOnNaN(),
#keras.callbacks.EarlyStopping(monitor='loss', patience=1),
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.backpropagation(X, Y,len(model.layers),10, start_time )
]
)

end_time = time.clock()
print("time",(end_time- start_time))

sys.exit(1)



# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

