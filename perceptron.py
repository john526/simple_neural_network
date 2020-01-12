import numpy as np


"""
READ README.rst

    Neural network example
    
    - inputs (data) : 
        - x1
        - x2
        - x3
    - synapses (way): 
       - w1
       - w2
       - w3
    - neurones : 
      - Σxi*wi = x1*w1+x2*w2+x3*w3 
         - i number(1,2,3,4,5,...)
         - xi number of data inputs
         - wi number of neurone
    - output :
      - y : result
      
    - phi (Greek alphabet):
      -  phi(x) = 1/1+e^(-x) | phi'(x) = x.(1-x)
      x =Σxiwi
      
      - phi(x) = 1/1+e^(-Σxiwi)
      
      - exp^(-x) : exponential de -x

"""

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivation(x):
    return x*(1-x)

# input
training_inputs = np.array([
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1]
])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)
synaptic_weights = 2* np.random.random((3,1))-1

print("Random starting synaptic weight : ")
print(synaptic_weights)

for iteration in range(20000): #1 after 20000 50000 100000 ...

    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer,synaptic_weights))

    error = training_outputs - outputs

    adjustements = error*sigmoid_derivation(outputs)

    synaptic_weights+=np.dot(input_layer.T, adjustements)

print("Synaptic weights after training ")
print(synaptic_weights)
print("\n")
print("Outputs after training : ")
print(outputs)