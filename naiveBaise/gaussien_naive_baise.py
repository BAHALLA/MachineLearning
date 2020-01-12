#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:44:22 2020

@author: bahalla
"""

import numpy as np
import pandas as pn
from statistics import variance
import math
import collections

dataset = pn.DataFrame({"Person":[0, 0, 0, 0, 1,1,1,1],
                        "height":[6, 5.92, 5.58, 5.92, 5,5.5,5.42,5.75],
                        "weight":[180, 190, 170, 165, 100,150,130,150],
                        "footsize":[12, 11, 12, 10, 6,8,7,9]})

class NBG:
    
    def __init__(self, dataset):
        print("--------NBG------------")
        self.data = dataset
        self.x_train = self.data.iloc[:,1:]
        self.y_train = self.data.loc[:,'Person']
        #counte number of occure of a class
        self.classes_dic = collections.Counter(self.y_train)
        #class's labels
        self.classes_labels = list(self.classes_dic.keys())
        #number of feature in dataset
        self.nbr_feaures = self.x_train.shape[1]  
        # number of classes
        self.nbr_classes = len(self.classes_dic) 
        #variance of features
    
    #pdf function 
    def pdf(self, x , m , v):
        return 1/(math.sqrt(2* math.pi* v)) * math.exp(-0.5*math.pow(x-m,2)/v)
    
    #return the probability of classes in dataset : p(ci)
    def probability_of_classes(self):
          
        nbr_elems = self.y_train.shape[0]
        cls_prob=np.ones(self.nbr_classes)
        #calculate the probability
        for i in range(self.nbr_classes):
            cls_prob[i] = self.classes_dic[i]/nbr_elems
      
        return cls_prob
    
    #calculate the mean and the variance
    def mean_variance(self):
        #initialize  m for means and v variance
        m= np.ones((self.nbr_classes,self.nbr_feaures))
        v= np.ones((self.nbr_classes,self.nbr_feaures))
    
        #extract feature by class and put it in list of features
        features = []
        for i in range(self.nbr_classes):
            features.append(self.data.loc[self.data['Person'] == self.classes_labels[i],:])
        
        #calculate mean and variance for every feature
        for j in range(len(features)):
            for k in range(self.nbr_feaures):
                col = features[j].iloc[:,k+1]
                m[j][k] = np.mean(col)
                v[j][k] = variance(col)
       
        return m , v
    
    #calculate the probablity of features p(x/c) for every class ci 
    def probabilty_features(self, m , v, t):
        
        probs = np.ones((self.nbr_classes, self.nbr_feaures))
        
        for i in range(self.nbr_classes):
            for j in range(self.nbr_feaures):
                probs[i][j] = self.pdf(t[j] , m[i][j] , v[i][j])
        
        return probs
    
    #Calculate the probabilty of classes
    def prediction_probs(self, m ,v ,t):
        probs = np.ones(self.nbr_classes)
        
        prb_cls = self.probability_of_classes()
        
        m,v = self.mean_variance()
    
        prb_feautres = self.probabilty_features(m,v,t)
        
        for i in range(self.nbr_classes):
            prod = 1
            for j in range(self.nbr_feaures):
                prod*= prb_feautres[i][j]
            prod*=prb_cls[i]
            probs[i] = prod
        return probs
                
#prediction    
nbg = NBG(dataset)
t= [6,130,8]
m,v= nbg.mean_variance()
probs = nbg.prediction_probs(m,v,t)
print(probs)
        