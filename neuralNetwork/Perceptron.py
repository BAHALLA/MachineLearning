#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:59:05 2020

@author: bahalla
"""


from numpy import exp, array, random

class Perceptron:
    
    def __init__(self, number_of_inputs):
        self.w = random.random_sample(number_of_inputs + 1)
        self.bias = float(-1)
        self.lr = 0.001
    #segmoid activation function
    def __segmoid(self, x):
        return 1/(1+ exp(-x))
    
    #step activation function
    def __step(self, x):
        if x >= 0:
            return 1
        else:
            return 0
    
    #predict function 
    def predict(self, inputs):
        
        result =0
        for i in range(len(inputs)):
            result += self.w[i] * inputs[i] 
        result += self.bias * self.w[-1]
        return self.__step(result)
    
    #train the modele
    def train(self, inputs, desired):
        
        predictedValue = self.predict(inputs)
        error = desired - predictedValue
        for i in range(len(inputs)):
            self.w[i] += inputs[i]*error*self.lr
        self.w[-1] += error*self.lr


#inputs = array([[6,180,12],[5.98,190,11],[5.58,170,12],[5.92,165,10],
         #       [5,100,6],[5.5,150,8],[5.42,130,7],[5.75,150,9]])
#desired = array([[0],[0],[0],[0],[1],[1],[1],[1]])
         
inputs = array([[0,0],[0,1],[1,0],[1,1]])
desired = array([[0],[0],[0],[1]])
p = Perceptron(4)
for i in range(len(inputs)):
        p.train(inputs[i], desired[i])

r =p.predict([0,1])
print("La resultat est :")
print (r)

