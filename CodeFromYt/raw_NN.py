#video:
# https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=3&ab_channel=sentdex

#Raw neural network
import numpy  as np 
import sys
import matplotlib

#example perceptron
inputs = [1.2, 5.1, 2.1] #the input layer (data you want to predict), layer 0

weights = [3.1, 2.1, 8.7] # weights of one neuron in layer 1
bias = 3 #bias of one neuron

output = inputs[0]*weights[0] + inputs[1]*weights[1] +inputs[2]*weights[2] + bias #output of neuron 1 in layer 1
print("Single neuron:", output)

inputs = inputs

weights = [[1, 2, 3],
		[1, 2, 2],
		[5, 4, 3]]

bias = [2,3,4]

layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, bias):
	neuron_output = 0
	for n_input, weight in zip(inputs, neuron_weights):
		neuron_output += n_input*weight
	neuron_output += neuron_bias
	layer_outputs.append(neuron_output)

for i in layer_outputs:
	print("{:.2f}".format(i), end = ', ')

print()

#############################
#LET'S DO IT WITH NUYMPY

inputs = np.array(inputs)
weights = np.array(weights)
bias = np.array(bias)

print(inputs.shape, weights.shape, bias.shape)

result = np.dot(weights,inputs) + bias

print(result)