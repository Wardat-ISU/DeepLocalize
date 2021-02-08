#!/usr/bin/env python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import sys
import keras
import time

NUM_TRAIN = 100000
NUM_TEST = 10000
INDIM = 3

mn = 1

def myrand(a, b) :
    return (b)*(np.random.random_sample()-0.5)+a

def get_data(count, ws, xno, bounds=100, rweight=0.0) :
    xt = np.random.rand(count, len(ws))
    xt = np.multiply(bounds, xt)
    yt = np.random.rand(count, 1)
    ws = np.array(ws, dtype=np.float)
    xno = np.array([float(xno) + rweight*myrand(-mn, mn) for x in xt], dtype=np.float)
    yt = np.dot(xt, ws)
    yt = np.add(yt, xno)

    return (xt, yt)


if __name__ == '__main__' :
    if 0 > 1 :
       EPOCHS = int(sys.argv[1])
       XNO = float(sys.argv[2])
       WS = [float(x) for x in sys.argv[3:]]
       mx = max([abs(x) for x in (WS+[XNO])])
       mn = min([abs(x) for x in (WS+[XNO])])
       mn = min(1, mn)
       WS = [float(x)/mx for x in WS]
       XNO = float(XNO)/mx
       INDIM = len(WS)
    else :
        INDIM = 3
        WS = [2.0, 1.0, 0.5]
        XNO = 2.2
        EPOCHS = 20

    X_test, y_test = get_data(10000, WS, XNO, 10000, rweight=0.4)
    X_train, y_train = get_data(100000, WS, XNO, 10000)

    model = Sequential()
    model.add(Dense(INDIM, input_dim=INDIM, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    start_time = time.clock()
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    model.fit(X_train, y_train, shuffle="batch", nb_epoch=EPOCHS,
      callbacks= [
#keras.callbacks.TerminateOnNaN()
#keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
keras.callbacks.DeepLocalize(X_train, y_train, len(model.layers), batch_size =32, startTime = start_time)
]        
              )
    endTime = time()
    score, acc = model.evaluate(X_test, y_test, batch_size=16)
    print (score)
    print (acc)

    predict_data = np.random.rand(100*100, INDIM)
    predictions = model.predict(predict_data)
    print("Time per image: {0} ".format(endTime-start_time)) 
    #for x in range(len(predict_data)) :
        #print ("%s --> %s" % (str(predict_data[x]), str(predictions[x])))