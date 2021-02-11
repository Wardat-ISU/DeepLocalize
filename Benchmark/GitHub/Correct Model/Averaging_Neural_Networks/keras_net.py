from keras.models import Sequential
from keras.layers import Dense, Dropout
import random
import numpy as np


# generates a large dataset of averages
def generate_dataset(values):
    dataset = []

    for _ in range(values):
        n1 = random.randint(0, 100)
        n2 = random.randint(0, 100)

        average = (n1 + n2) / 2

        dataset.append([n1, n2, average])

    return np.array(dataset)


# scale
def preprocess(data):
    return data/100


# unscale
def postprocess(data):
    return data * 100

training_data_length = 10000
BATCH_SIZE = 10
EPOCHS = 10

training_data = generate_dataset(training_data_length)
split_data = np.split(training_data, [2, ], axis=1)
training_input = preprocess(split_data[0])
training_output = preprocess(split_data[1]).flatten()

model = Sequential()
model.add(Dense(2, input_dim=training_input.shape[1]))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')

model.fit(training_input, training_output, epochs=EPOCHS, batch_size=BATCH_SIZE)

prediction = model.predict(preprocess(np.array([[10, 20]])))
print(postprocess(prediction.flatten()[0]))
