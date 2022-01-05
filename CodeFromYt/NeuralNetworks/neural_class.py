import numpy as np

np.random.seed(0)

X = [[1,2,3,2.5],
	[2.0, 5.0, 1.0, 2.0],
	[-1.5, 2.7, 3.3, -0.8]]

class Acrivation_ReLU: #Rectified Linear Unit
	def forward(self, inputs):
		self.output = np.maximum(0,inputs)

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases



#randn return a gaussian distribution bounded around zero
layer_1 = Layer_Dense(4,5) #since my inputs (X) has 4 inputs, and 5 is your choice
layer_2 = Layer_Dense(5,2) # since 5 outputs from layer 1, these will be the inputs of layer 2

layer_1.forward(X)
print(layer_1.output)
layer_2.forward(layer_1.output)
print(layer_2.output)

#activation functions
# we prefer the ReLU instead of the sigmoid since it is easy and fast.
#but why do we actually use the act. function?
#not using an act. funct? y = x. whatever the input, that's the output
#we need some non linear stuff to predict for example a cosine distribution
#adding a little non linearity (like sigmoind, or rectified linear) makes a huge difference 



