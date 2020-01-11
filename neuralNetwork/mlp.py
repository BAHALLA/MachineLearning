
"""
Created on Fri Dec 20 09:33:20 2019

@author: bahalla
"""

import numpy as np
#import math

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def loss(desired, predicted):
    return 1/2*(desired - predicted)**2

#Inputs
inputs = np.array([[0.1,0.2]])
desired_output = np.array([[0.03]])
learning_rate = 0.1

#weights and bias initialization
hidden_weights = [[0.5,-0.7],[0.6,0.8]]
hidden_bias = [[0.4,0.9]]
output_weights = np.array([[0.35],[-0.45]])
output_bias = np.array([[0.25]])
print("-------------------------------------------------------------------")


i = 0
for i in range(1000):    
    #Learning process
    #Forward Pass
    hidden_layer_in = np.dot(inputs,hidden_weights) + hidden_bias
    hidden_layer_out = sigmoid(hidden_layer_in)
    output_layer_in = np.dot(hidden_layer_out,output_weights) + output_bias
    predicted_output = sigmoid(output_layer_in)
    
    #Backward pass
    error = desired_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_out)
    
    #Updating Weights and Biases
    output_weights += hidden_layer_out.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * learning_rate
    hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * learning_rate

print("prediected value :", predicted_output)
#loss_=loss(predicted_output,desired_output)
#print("Loss :", loss_)
#print("updated hidden weights: ")
#print(hidden_weights)
#print("updated hidden biases: ")
#print(hidden_bias)
#print("updated output weights: ")
#print(output_weights)
#print("updated output biases: ")
#print(output_bias)

      