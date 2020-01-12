#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 23:25:02 2020

@author: bahalla
"""
import numpy as np

class NN:
    
    def __init__(self, input_layer, hidden_layer, output_layer):
        #initaite weights
        self.w_hidden_layer = np.array([[0.5,-0.7],[0.6,0.8]])#np.random.random((input_layer, hidden_layer)) 
        self.w_output_layer = np.array([[0.35],[-0.45]])#np.random.random((hidden_layer, output_layer)) 
        #bais per layer
        self.hidden_layer_bias = np.array([[0.4,0.9]]) #np.random.random((1,hidden_layer)) 
        self.ouput_layer_bais = np.array([[0.25]]) #np.random.random((1,output_layer)) 
        
        #learning rate
        self.learning_rate = 0.1
    
    #Acivation function segmoid
    def __sedmoid(self, x):
         return 1/(1+ np.exp(-x))
     
    #derivative function of segmoid
    def __segmoid_derivative(self, x):
        return x*(1-x)
    
    #predict function
    def predict(self, inputs):
        hidden_layer_out = self.__sedmoid(np.dot(inputs, self.w_hidden_layer) + self.hidden_layer_bias)
        output_layer_out = self.__sedmoid(np.dot(hidden_layer_out, self.w_output_layer) + self.ouput_layer_bais)
        return hidden_layer_out, output_layer_out
    
    #display weights
    def print_weights(self):
       print(self.w_output_layer)
    #train the modele
    def train(self, inputs, desired_output, iterations):
        i=0
        for i in range(iterations):
            #predict 
            hidden_layer_out, output_layer_out = self.predict(inputs)
            
           
            #error in output layer
            error_output = desired_output - output_layer_out
            deltat_predicted_output = error_output * self.__segmoid_derivative(output_layer_out)
            #error in hidden layer
            error_hidden = deltat_predicted_output.dot(self.w_output_layer.T)
            deltat_hidden_layer = error_hidden * self.__segmoid_derivative(hidden_layer_out)
            
             #Updating Weights and Biases
            self.w_output_layer += hidden_layer_out.T.dot(deltat_predicted_output) * self.learning_rate
            self.ouput_layer_bais += np.sum(deltat_predicted_output,axis=0,keepdims=True) * self.learning_rate
            self.w_hidden_layer += inputs.T.dot(deltat_hidden_layer) * self.learning_rate
            self.hidden_layer_bias += np.sum(deltat_hidden_layer,axis=0,keepdims=True) * self.learning_rate
            
#train and use the modele
nn = NN(2,2,1)
inputs = np.array([[0.1,0.2]])
desired_output = np.array([[0.03]])
nn.train(inputs,desired_output,10000)
print(nn.predict(inputs))



