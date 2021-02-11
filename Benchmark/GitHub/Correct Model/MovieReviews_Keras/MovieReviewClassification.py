import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

def vectorize_sequences(sequences, dimensions=10000) :
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences) :
        results[i, sequence] = 1.
    return results

print('____Preparing data____')

import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# restore np.load for future normal usage
np.load = np_load_old

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print('____Building model____')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

print('____Training model____')

history = model.fit(x_train[10000:],
                    y_train[10000:],
                    epochs=7,
                    batch_size=512,
                    validation_data=(x_train[:10000], y_train[:10000]))

print('____Evaluating model____')

evaluation_results = model.evaluate(x_test, y_test)
print(evaluation_results)

print('____Predicting____')

prediction_results = model.predict(x_test)
print(prediction_results)

#history_dict = history.history
#loss_values = history_dict['loss']
#val_losts_values = history_dict['val_loss']

#epochs = range(1, len(loss_values) + 1)

#plt.plot(epochs, loss_values, 'bo', label='Training loss')
#plt.plot(epochs, val_losts_values, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()

#plt.show()

#plt.clf()
#acc_values = history_dict['binary_accuracy']
#val_acc_values = history_dict['val_binary_accuracy']

#plt.plot(epochs, acc_values, 'bo', label='Training acc') 
#plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()

#plt.show()
