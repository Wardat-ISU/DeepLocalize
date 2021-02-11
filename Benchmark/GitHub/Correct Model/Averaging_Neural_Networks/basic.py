import numpy as np
import random

#####################################################
# Neural network with two input nodes, one hidden
# layer with two neurons, and an output neuron.
# Calculates the average of the input values.

#####################################################
# relu
def activation(x):
    if x > 0:
        return x
    else:
        return 0


# relu derivative
def d_activation(x):
    if x > 0:
        return 1
    else:
        return 0


# single forward propagation
def forward_prop(input, weight1, weight2):
    hlayer = np.dot(input, weight1)[0]
    hlayer = np.array([[activation(hlayer[0]), activation(hlayer[1])]])

    output = activation(np.dot(hlayer, weight2))

    return hlayer, output


# single backward propagation
def back_prop(input, output, weight1, weight2, hidden_layer, true_output):
    current_weight2 = weight2

    # weight2 back propagations
    error = true_output - output

    # caps the value of error to prevent inf values
    if abs(error) > 0.01:
        if error > 0:
            error = 0.01
        else:
            error = -0.01

    delta2 = error * hidden_layer.T
    weight2 += delta2

    # weight1 back propagation
    delta_matrix = np.hstack((current_weight2, current_weight2))
    delta1 = input.T * delta_matrix * error

    weight1 += delta1

    return weight1, weight2


# trains the neural net on a given dataset
def train(data):
    weight1 = np.random.rand(2, 2)
    weight2 = np.random.rand(2, 1)

    for set in data:
        training_output = set[-1]
        set = np.array([set[:-1]])

        hidden, out = forward_prop(set, weight1, weight2)
        weight1, weight2 = back_prop(set, out, weight1, weight2, hidden, training_output)

    return weight1, weight2


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


# generates training dataset
training_data = generate_dataset(10000)
scaled = preprocess(training_data)

# preprocess for scaling. probably divide by 100

weight1, weight2 = train(scaled)

# input for predicting. Modify values to test averaging
prediction_input = preprocess(np.array([[0, 1]]))
_, prediction = forward_prop(prediction_input, weight1, weight2)

print(postprocess(prediction[0][0]))



