import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
import math
import time
from sklearn.preprocessing import StandardScaler

x = np.arange(-100, 100, 0.5)
y = x**4


x_scaler = StandardScaler()
y_scaler = StandardScaler()

x = x_scaler.fit_transform(x[:, None])  # Features are expected as columns vectors.
y = y_scaler.fit_transform(y[:, None])
model = Sequential()
model.add(Dense(50, input_shape=(1,)))
model.add(Activation('relu'))
model.add(Dense(50) )
model.add(Activation('elu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

t1 = time.clock()
for i in range(100):
    model.fit(x, y, epochs=1000, batch_size=len(x), verbose=1)
    predictions = model.predict(x)
    print (i," ", np.mean(np.square(predictions - y))," t: ", time.clock()-t1)

    #plt.hold(False)
    plt.plot(x, y, 'b', x, predictions, 'r--')
    plt.hold(True)
    plt.ylabel('Y / Predicted Value')
    plt.xlabel('X Value')
    plt.title([str(i)," Loss: ",np.mean(np.square(predictions - y))," t: ", str(time.clock()-t1)])
    plt.pause(0.001)

#plt.savefig("fig2.png")
plt.show()