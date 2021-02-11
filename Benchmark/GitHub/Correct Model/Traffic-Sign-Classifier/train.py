# Python Standard Library
from datetime import datetime
import json
import os
import time

# Public Libraries
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Lambda, Dropout, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

# Project
import config
import helper

DATETIME_FORMAT = '%y-%m-%d_%H-%M'

# Overwrite model files
overwrite_model = True

# Data set ('test', 'train', 'valid')
data_set = 'train'

# Load images and labels
x, y = helper.load_data(data_set)

# Select subsample for faster debugging
sample_fraction = 1
num_samples = x.shape[0]
sample_size = round(sample_fraction * num_samples)
x = x[:sample_size]
y = y[:sample_size]

# Split into training and validation
test_fraction = 0.20
x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=0,
                                                  test_size=test_fraction)

# Hyperparams
shape = x.shape[1:]
num_classes = config.NUM_CLASSES
learning_rate = 0.001
batch_size = 512
epochs = 10

# Class number to classification columns (categorical to dummy variables)
y_train = np_utils.to_categorical(y_train, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)

# Model of Convolutional Neural Network
model = Sequential()
model.add(Lambda(lambda p: p/255.0 - 0.5, input_shape=shape))
model.add(Conv2D(3, (1, 1), activation='sigmoid'))
model.add(Conv2D(16, (5, 5), strides=(2, 2), activation='elu'))
model.add(Conv2D(32, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Print model summary
model.summary()

# Compile model
model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy',
              metrics=['accuracy'])

# Start training train
start_time = time.time()

# Configure Tensorboard log
timestamp = datetime.fromtimestamp(start_time).strftime(DATETIME_FORMAT)
log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, timestamp)
tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                         write_images=True)

# Train model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                    verbose=2, validation_data=(x_val, y_val),
                    callbacks=[tbCallBack])

# Training duration
training_time = time.time() - start_time

# Print metrics of validation set
print('')
print('*** Training Complete ***')
print('Elapsed time: %.1f seconds' % training_time)
scores = model.evaluate(x_val, y_val, verbose=0)
names = model.metrics_names
print('')
print('*** Metrics ***')
for name, score in zip(names, scores):
    print('%s: \t%.4f' % (name, score))

# Overwrite saved model
if overwrite_model:
    model.save_weights(config.MODEL_WEIGHTS, overwrite=True)
    with open(config.MODEL_DEFINITION, 'w') as outfile:
        json.dump(model.to_json(), outfile)
    plot_model(model, config.MODEL_DIAGRAM, show_shapes=True)
    print('')
    print('*** Model Saved ***')
