#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:36:07 2019

@author: bahalla
"""

from numpy import exp, array, dot, random

# Neuron Layer
class NL:
    #define a layer with numer of neurons and inputs
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.layer_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons))-1

# Neural Network
class NN:
    
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
        self.learning_rate = 0.1
    
    # segmoid activation function normlize outputs between 0 and 1
    def __segmoid(self, x):
        return 1/(1+ exp(-x))
    
    #segmoid derivative function for error propagation
    def __segmoid_derivative(self, x):
        return x * (x -1)
    
    #prediction function return the output from layer 1 and 2
    def predict(self, inputs):
        output_layer1 = self.__segmoid(dot(inputs, self.layer1.layer_weights))
        output_layer2 = self.__segmoid(dot(output_layer1, self.layer2.layer_weights))
        return output_layer1, output_layer2
    
    #train of the model by trying to predict and adjust weights
    def train(self, inputs, outputs, iterations):
        for iteration in range(iterations):
            output_layer1, output_layer2 = self.predict(inputs)
            
            #calculate error in layer 2
            layer2_error = outputs - output_layer2
            layer2_deltat = layer2_error * self.__segmoid_derivative(output_layer2)
            
            #calculate error in layer 1
            layer1_error = layer2_deltat.dot(self.layer2.layer_weights.T)
            layer1_deltat = layer1_error * self.__segmoid_derivative(output_layer1)
            
            #Adjust weights
            self.layer1.layer_weights += inputs.T.dot(layer1_deltat) * self.learning_rate
            self.layer2.layer_weights += output_layer1.T.dot(layer2_deltat) * self.learning_rate
   
    def print_weights(self):
        print ("Layer 1 weights :")
        print(self.layer1.layer_weights)    
        print ("Layer 2 weights :")
        print(self.layer2.layer_weights)

if __name__ == "__main__":
    random.seed(1)
    layer1 = NL(2,2)
    layer2 = NL(1,2)
    
    nn = NN(layer1, layer2)
    #nn.print_weights()
    
    inputs = array([[0,0],[0,1],[1,0],[1,1]])
    outputs = array([[0],[1],[1],[0]]).T
    nn.print_weights()
    nn.train(inputs, outputs, 1000)
   
    
    hidden , out =nn.predict(array([1,0]))
    print(out)
          
        