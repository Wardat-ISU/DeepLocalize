# DeepLocalize: Fault Localization for Deep Neural Networks


## Our Tool

The simplest way to build a training model is to start with the [`Sequential`] model, our tool follows the [Keras functional API](https://keras.io/getting-started/functional-api-guide), with some changes explained in the paper:

Here is the `Sequential` model:

```python
model = Sequential()
```

Add all layers as easy as by using `.add()` function.

Dense layer takes three arguments: num_inputs(number of input unit), num_outputs(number of output unit), lr_rate(leaning rate), and name(name of layer).

The activation function is added as a new layer. 
```python
lr = 0.01 
model.add(Dense(num_inputs=100, num_outputs=64, lr_rate=lr, name='FC1'))
model.add(ReLu())
model.add(Dense(num_inputs=64, num_outputs=10, lr_rate=lr, name='FC2'))
model.add(softmax())
```


Once you finished building your model, you can use `.compile()` to start the learning process after determing the loss function, optimizer and the metrics:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

The core principle of our tool is to make the training model simple, while inserting instrumentation
in the `.fit()` function to observe the model variables, then make the user to be fully in control when they need the variables. 

```python
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

You can start training a new model using any environment for scientific programming in the Python language (Windows) or commands in Terminal (macOS/Linux) as following:

    python main.py


## Our Callback Function
To use our callback, you need to add our callback as subclass in your keras.callbacks.py file.

The core principle of our callback to get a view on internal states and statistics of the model during training.

Then you can pass our callback `DeepLocalize()` to the `.fit()` method of a model as following:

```python
callback = keras.callbacks.DeepLocalize(inputs, outputs, layer_number, batch_size, startTime)
model = keras.models.Sequential()
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation(activations.relu))
model.compile(keras.optimizers.SGD(), loss='mse')
model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10, batch_size=1, 
...                     callbacks=[callback], verbose=0)
```



## Prerequisites

Version numbers below are of confirmed working releases for this project.

    python 3.6.5
    keras 2.2.2   
    numpy 1.14.3
    pandas 0.23.4
    scikit-learn 0.19.1
    scipy 1.1.0
    tensorflow 1.10.1

## BibTeX Reference
If you find this [paper](https://conf.researchr.org/details/icse-2021/icse-2021-papers/1/DeepLocalize-Fault-Localization-for-Deep-Neural-Networks) useful in your research, please consider citing:

    @inproceedings{wardat21@DeepLocalize,
	Author = {Mohammad Wardat and Wei Le and Hridesh Rajan},
	Title = {DeepLocalize: Fault Localization for Deep Neural Networks},
	Booktitle  = {ICSE'21: The 43nd International Conference on Software Engineering},
	Year = {2021},
	entrysubtype = {conference}
    }
