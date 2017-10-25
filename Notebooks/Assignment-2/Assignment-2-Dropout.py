import tensorflow as tf
import numpy as np
from random import seed
from random import random
from random import randint
from math import exp
import sklearn as skl
from sklearn.metrics import f1_score
#from sklearn import f1_score




def initialize_network(n_inputs, n_neurons , n_outputs, dropout_prob):
	network = list()
	hidden_layer1 = [{'weights' :generate_layer(n_inputs),'dropped': decision(0)} for i in range(n_neurons)]
	network.append(hidden_layer1)
	hidden_layer2 = [{'weights' :generate_layer(n_neurons),'dropped': decision(dropout_prob)} for i in range(n_neurons)]
	network.append(hidden_layer2)
	output_layer = [{'weights' :generate_layer(n_neurons),'dropped': decision(0)} for i in range(n_outputs)]
	network.append(output_layer)
	return network

def generate_layer(nInputs):
    layer = [random() for layer in range(nInputs)]
    return layer

def decision(probability):
    return random() < probability

# Calculate neuron activation for an input
def weights_input_product(weights, inputs):
	summation = 0
	for i in range(len(weights)-1):
		summation += weights[i] * inputs[i]
	return summation

def sigmoid(z):
	return 1.0 / (1.0 + exp(-z))

#Send the list of outputs for each layer
def forward_propagate(network, inputData):
    outputs = []
    inputRecord = inputData
    for layer in network: # Iterate over the layers
        layer_output = []
        i = 0
        for neuron in layer: # Iterate for all neurons
            if neuron['dropped']:
                neuron['weights'] = np.zeros(len(neuron['weights']))
            summation = weights_input_product(neuron['weights'],inputRecord)
            if i < len(network):
                activation = sigmoid(summation)
            else:
                activation = softmax(summation)
            if not neuron['dropped']:
                neuron['output'] = activation
            else:
                neuron['output'] = 0
            layer_output.append(activation)
        outputs.append(layer_output)
        inputRecord = layer_output
    return layer_output

def softmax(z):
    sum = np.sum(np.exp(z), axis=1, keepdims=True)
    return np.divide(np.exp(z),sum)

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

def transfer_softmax_derivative(signal):
    return signal*(1-signal) + (1 - signal)*signal


def transfer_softmax_derivative1(signal):
    return np.multiply( signal, 1 - signal ) + sum(
            # handle the off-diagonal values
            - signal * np.roll( signal, i, axis = 1 )
            for i in xrange(1, signal.shape[1] )
        )

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                #print('For Layer:',j)
                for neuron in network[i + 1]:
                    if not neuron['dropped']:
                        error += (neuron['weights'][j] * neuron['delta'])
                    #print('Neuron dropped:', neuron['dropped'],'Error:',error)
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                if not neuron['dropped']:
                    errors.append(expected[j] - neuron['output'])
                else:
                    errors.append(expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            if not neuron['dropped']:
                if j != len(network) - 1:
                    neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])  # Sigmoid derivative for all other layers
                else:
                    neuron['delta'] = errors[j] * transfer_softmax_derivative(neuron['output'])#softmax derivative for the output layer
            else:
                neuron['delta'] = 0.0
# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                if not neuron['dropped']:
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            if not neuron['dropped']:
                neuron['weights'][-1] += l_rate * neuron['delta']

#Train via SGD
def train_network(networks, train, l_rate, n_epoch, n_outputs,n_iterations):
	for epoch in range(n_epoch):
		sum_error = 0
		n_examples = len(train)
        for iteration in range(n_iterations):
            random_samples = train[np.random.choice(train.shape[0], 1, replace=False), :]
            networkToPick = randint(0, len(networks) - 1)
            print('Network Index Picked:',networkToPick)
            random_network = networks[networkToPick]
            for row in random_samples:
                outputs = forward_propagate(random_network, row)
                expected = [0 for i in range(n_outputs)]
                #print("Expected shape",expected)
                expected[int(row[-1])] = 1
                #expected[1] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                backward_propagate_error(random_network, expected)
                update_weights(random_network, row, l_rate)
                print('Epoch=%d, Loss=%.3f' % (epoch, sum_error))



def softmax(z):
    sum = np.sum(np.exp(z))
    return np.divide(np.exp(z),sum)

def get_weighted_network(networks, dropout_prob):
    network = list()
    hidden_layer1 = [{'weights' :generate_weighted_weights(networks,0,dropout_prob,n_inputs, i),'dropped':False} for i in range(n_neurons)]
    network.append(hidden_layer1)
    hidden_layer2 = [{'weights' :generate_weighted_weights(networks,1,dropout_prob,n_neurons, i),'dropped':False}  for i in range(n_neurons)]
    network.append(hidden_layer2)
    output_layer = [{'weights' :generate_weighted_weights(networks,2,dropout_prob,n_neurons, i),'dropped':False} for i in range(n_outputs)]
    network.append(output_layer)
    return network

def generate_weighted_weights(networks, layerIndex, dropout_prob, rowCount, columnCount):
    weights = np.zeros(rowCount)
    for network in networks:
        layer = network[layerIndex]
        neuron = layer[columnCount]
        val = [(1 - dropout_prob) * neuron['weights'][i]  for i in range(rowCount)]
        weights += val
    return weights

def predict(network, row):
	p=outputs = forward_propagate(network, row)
	#print(outputs[0],outputs[1])
	#print(np.asarray(outputs).T)
	return outputs.index(max(outputs))

#Load Dataset
inputFilenameWithPath = 'train_data.txt'
inputData = np.loadtxt(inputFilenameWithPath, delimiter=",")
n_inputs = len(inputData[0]) - 1
n_outputs = len(set([row[-1] for row in inputData]))
n_neurons = 3
dropout_prob = 0.3

#Testing code
networks = []
network1 = initialize_network(n_inputs, n_neurons, n_outputs, dropout_prob)
networks.append(network1)
networks.append(initialize_network(n_inputs, n_neurons, n_outputs, dropout_prob))
networks.append(initialize_network(n_inputs, n_neurons, n_outputs, dropout_prob))
networks.append(initialize_network(n_inputs, n_neurons, n_outputs, dropout_prob))

sgd_learning_rate = 0.1
numberOfEpochs  = 50
numberOfExamplesPerEpoch = 600

train_network(networks, inputData, sgd_learning_rate, numberOfEpochs, n_outputs,numberOfExamplesPerEpoch)

for network in networks:
    for layer in range(len(network)):
        print('Layer',layer)
        for neuron in network[layer]:
            print(neuron['dropped'])
            print(neuron['weights'])
weightedNetwork = get_weighted_network(networks, dropout_prob)


print('Weighted Network')
for layer in range(len(weightedNetwork)):
        print('Layer',layer)
        for neuron in network[layer]:
            print(neuron['dropped'])
            print(neuron['weights'])

predictions = []
truth = inputData[:,2]
for row in inputData:
	prediction = predict(weightedNetwork, row)
	predictions.append(prediction)
f1 = skl.metrics.f1_score(truth, predictions, average='micro')
precision = skl.metrics.precision_score(truth, predictions, average='micro')
recall = skl.metrics.recall_score(truth, predictions, average='micro')
print('\nTraining Precision=%.2f' % (precision))
print('Training Recall=%.2f' % (recall))
print('Training F1 Score=%.2f' % (f1))

print('\n---------------------------- Testing the predictions -------------------------')
# Test making predictions with the network
inputFilenameWithPath = 'test_data.txt'
dataset = np.loadtxt(inputFilenameWithPath, delimiter=",")

#predictions = prediction_op(dataset[:,:2])
truth = dataset[:,2]
predictions = []
for row in dataset:
	prediction = predict(networks[0], row)
	predictions.append(prediction)
	#print('Expected=%d, Got=%d' % (row[-1], prediction))
f1 = skl.metrics.f1_score(truth, predictions, average='micro')
precision = skl.metrics.precision_score(truth, predictions, average='micro')
recall = skl.metrics.recall_score(truth, predictions, average='micro')
print('\nTest Precision=%.2f' % (precision))
print('Test Recall=%.2f' % (recall))
print('Test F1 Score=%.2f' % (f1))