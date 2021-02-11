"""
Training neural network to play Conway's Game of Life
"""

import numpy as np
#import tensorflow as tf
import sys
#import tensorflow as tf
import  keras
from keras.models import Sequential

#from tensorflow.keras import optimizers



from time import *



def generate_field(shape, alive_prob=0.5):
    return (np.random.rand(*shape) < alive_prob).astype(np.int32)


def update_field(field, hw_axis=(1, 2)):
    """
    Calculates next state for the cell field according to the rules.

    Params:
    -------

    field: ndarray, usually of shape (N, H, W, 1), N-num of samples, H-height, W-width
    hw_axis: tuple with the order of height and width dimension in field shape.

    Returns:
    --------

    ndarray: field of the same shape for the next state.
    """
    h_ax, w_ax = hw_axis

    neighbours = np.zeros(field.shape, dtype=np.int32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if not (dx or dy):
                continue
            neighbours += np.roll(np.roll(field, dx, w_ax), dy, h_ax)

    return np.logical_or(
        neighbours==3,
        np.logical_and(field, neighbours==2)
    ).astype(np.int32)


def pad_field(field):
    """
    Pads H and W dimensions to emulate cyclic nature of cell field.

    Params:
    -------

    field: ndarray (N, H, W, 1), N-num of samples, H-height, W-width

    Returns:
    --------

    ndarray (N, H+2, W+2, 1)
    """
    return np.pad(field, ((0,0), (1,1), (1,1), (0,0)), mode='wrap')


def generate_dataset(num_samples, height, width, alive_prob=0.5):
    """
    Generates dataset with shape (num_samples, height, width, 1) where each
    cell is alive with alive_prob.

    Returns:
    --------

    tuple (X, y)
    X: ndarray (N, H, W, 1) field states
    y: ndarray (N, H, W, 1) corresponding next states
    """
    shape = (num_samples, height, width, 1)
    X = generate_field(shape, alive_prob)
    y = update_field(X)
    return X, y


def life_nn(
        height,
        width,
        num_filters=10,
        num_channels=20,
        loss='binary_crossentropy',
        optimizer='adam'#keras.optimizers.Adam(learning_rate=0.01)
    ):
    """
    Create CNN model for the Game of Life.

    In game's rules next state of a cell depends only on its neighbours.

    Thats why (3,3) conv filters is a reasonable choice for the first layer.
    For this layer 'valid' padding is used to shrink shape to the original size
    of the field.

    Then (1,1) convolution layer with num_channels filters is used.

    And finally another (1,1) convolution with just 1 output channel.

    (N, H+2, W+2, 1)
           |
        conv1 (3,3)
           |
    (N, H, W, num_filters)
           |
        conv2 (1,1)
           |
    (N, H, W, num_channels)
           |
        conv3 (1,1)
           |
    (N, H, W, 1)
    """
    model = Sequential()
    model.add( keras.layers.Conv2D(
            num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            use_bias=True,
            input_shape=(height+2, width+2) + (1,),
            #name='conv1_3x3'
        ))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(
            num_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            #activation='relu',
            #name='conv2_1x1'
        )) 
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(
            1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            #activation='sigmoid',
            #name='conv3_1x1'
        ))
    model.add(keras.layers.Activation('sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def train(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=1):
    """
    Trains NN model on train and validation set.
    Wrap pads input fields to prevent errors on the border.
    """
    X_train_padded = pad_field(X_train)
    X_val_padded = pad_field(X_val)
    model.fit(X_train_padded, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_padded, y_val), callbacks= [
            #keras.callbacks.TerminateOnNaN(),
            #keras.callbacks.EarlyStopping(monitor='loss', patience=1),
            #keras.callbacks.EarlyStopping(monitor='accuracy', patience=1),
            keras.callbacks.backpropagation(X_train_padded, y_train, len(model.layers), batch_size, start_time)
            ]

    )


def evaluate(model, X_test, y_test, batch_size=64):
    """
    Evaluates model's loss and accuracy on the test set.
    Wrap pads input fields before the evaluation.
    """
    X_test_padded = pad_field(X_test)
    return model.evaluate(
        X_test_padded,
        y_test,
        batch_size=batch_size,
        verbose=False
    )


def evaluate_prob_grid(model):
    """
    Generates test dataset for different alive probabilities
    and checks model performance.
    """
    shape = model.input_shape
    height, width = shape[1] - 2, shape[2] - 2
    for alive_prob in np.linspace(0.1, 0.9, 9): 
        X_test, y_test = generate_dataset(1000, height, width, alive_prob)
        loss, acc = evaluate(model, X_test, y_test)
        print('P_alive={:.1f} Loss:{:.2f} Acc:{:.2f}'.format(alive_prob, loss, acc))


def print_evolution(height, width, alive_prob=0.5, epochs=20):
    """
    Visualize field evolution for debugging purposes.
    """
    import time
    field = generate_field((height, width))

    def print_field(field):
        for row in field:
            print(''.join('X' if x else '.' for x in row))

    for epoch in range(epochs):
        print('Epoch:', epoch)
        print_field(field)
        print()
        field = update_field(field, hw_axis=(0,1))
        time.sleep(0.3)


if __name__ == '__main__':
    import sys
    #height = int(sys.argv[1])
    #width = int(sys.argv[2])
    
    height = 3
    width = 3
    start_time = time()
    print('Building model:')
    model = life_nn(height, width)
    model.summary()
    print()

    num_train_samples = 8000
    num_val_samples = 2000
    X_train, y_train = generate_dataset(num_train_samples, height, width)
    X_val, y_val = generate_dataset(num_val_samples, height, width)

    print('Training model:')
    train(model, X_train, y_train, X_val, y_val, epochs=2)
    print()
    end_time = time()
    print("time",(end_time- start_time))

    sys.exit(1)


    print('Evaluating model:')
    evaluate_prob_grid(model)
