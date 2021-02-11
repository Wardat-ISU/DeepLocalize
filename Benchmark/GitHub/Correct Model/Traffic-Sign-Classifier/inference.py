# Python Standard Library
import json

# Public libraries
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.utils import np_utils

# Project
import config
import helper


DATA_SET = 'test'  # {'test', 'train', 'valid'}

# Load images and labels
x, y = helper.load_data(DATA_SET)

# Load model
model_file = config.MODEL_DEFINITION
with open(model_file, 'r') as jfile:
    model = model_from_json(json.loads(jfile.read()))

# Compile model and load weights
model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=['accuracy'])
model.load_weights(config.MODEL_WEIGHTS)

# Evaluate model performace
print('Evaluating performance on %d samples' % x.shape[0])
y_cat = np_utils.to_categorical(y, config.NUM_CLASSES)
scores = model.evaluate(x, y_cat, verbose=0)
names = model.metrics_names
for name, score in zip(names, scores):
    print('%s: \t%.4f' % (name, score))
