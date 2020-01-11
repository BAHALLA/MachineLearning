#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 00:02:30 2019

@author: bahalla
"""
import numpy as np
import random

def segmoid(x):
    return 1/(1+np.exp(-x))
def segmoid_derv(x):
    return x*(x-1)

class NN:
    
    def __init__(self, inputs):
        random.seed(1)
        self.inputs= inputs
        self.l =  len(inputs)
        self.li = len(inputs[0])
        
        self.wi = np.random.random((self.li,self.l))
        self.wh = np.random.random((self.l, 1))
        
    def predict(self, inputs):
        s1 = segmoid(np.dot(inputs, self.wi))
        s2 = segmoid(np.dot(s1, self.wh))
        return s2
    def train(self, inputs, outputs, it):
        
        for i in range(it):
            l0= inputs
            l1= segmoid(np.dot(l0 , self.wi))
            l2= segmoid(np.dot(l1, self.wh))
            
            l2_error = outputs - l2
            l2_d = np.multiply(l2_error, segmoid_derv(l2))
            
            l1_error = np.dot(l2_d, self.wh.T)
            l1_d = np.multiply(l1_error, segmoid_derv(l1))
            
            self.wh += np.dot(l1.T, l2_d)
            self.wi += np.dot(l0.T, l1_d)

inputs = np.array([ [0,0], [0,1] , [1,0] , [1,1]])
outputs = np.array([[0], [1], [1], [0]])

nn = NN(inputs)
print ("before training")
dd =nn.predict(inputs)
print(dd)
nn.train(inputs, outputs, 100000)
print ("after training")
rr=nn.predict(inputs)
print(rr)

            
            
        
    
        