import numpy as np

#Batch size, usually 32 is top
#what's the batch size? How many inputs you pass to the network at a time.
#Too big batch size -> overfitting 


inputs = [[1, 2, 3, 2.5],
			[2.0, 5.0, -1.0, 2.0],
			[-1.5, 2.7, 3.3, -0.8]]
weights = [[ 0.2, 0.8, -0.5, 1.0],
			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

inputs = np.array(inputs)
weights = np.array(weights)

#output = np.dot(inputs,weights) + biases
#print(output)
#This will give you "ValueError: shapes (3,4) and (3,4) not aligned: 4 (dim 1) != 3 (dim 0)"
#Because the dot product tries to do: 1*0.2 + 2*0.5 + 3*-0.26 + 2.5*????
#You need to take weights and switch rows and columns -> Transpose matrix :) 

weights = weights.transpose()

output = np.dot(inputs,weights) + biases
print(output)

# Now it works :)