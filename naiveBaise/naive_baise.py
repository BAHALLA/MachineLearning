#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:47:36 2019

@author: bahalla
"""

import numpy as np
import collections
import pandas as pn
import math
from statistics import variance

dataset = pn.DataFrame({"Person": [0,0,0,0,1,1,1,1],
                        "height":[6,5.98,5.58,5.92,5,5.5,5.42,5.75],
                        "weight":[180,190,170,165,100,150,130,150],
                        "footsize":[12,11,12,10,6,8,7,9]})

def pdf(x,m,v):
    return 1/math.sqrt(2*math.pi*v)*math.exp(-0.5*math.pow(x-m,2)/v)

def split(dataset):
    x=dataset.iloc[:,1:]
    y=dataset.loc[:,'Person']

    return x,y


x,y = split(dataset)
def classes_probability(y):
    y_dict = collections.Counter(y)
    n_classes = len(y_dict)
    prob=np.ones(n_classes)
    nb_elem = y.shape[0]
    for i in range(n_classes):
         prob[i] = y_dict[i]/nb_elem
     
    return  prob
def mean_var(dataset):
#m = moyenne and v = variance
    y_dict = collections.Counter(y)
    n_classes = len(y_dict)
    n_cols = dataset.shape[1]-1
    m= np.ones((n_classes,n_cols))
    v= np.ones((n_classes,n_cols))
    
    males = dataset.loc[dataset['Person'] == 0,:]
    famles = dataset.loc[dataset['Person'] == 1,:]
    
    for j in range(n_cols):
            col = males.iloc[:,j+1]
            m[0][j] = np.mean(col)
            v[0][j] = variance(col)
    for j in range(n_cols):
            col = famles.iloc[:,j+1]
            m[1][j] = np.mean(col)
            v[1][j] = variance(col)
    
    return m,v

def probability_features_classes(m,v,sample):
    n_classes =m.shape[0]
    n_cols = m.shape[1]
    probs=np.ones((n_classes,n_cols))
    
    for i in range(n_classes):
        for j in range(n_cols):
            probs[i][j] = pdf(sample[j], m[i][j],v[i][j])
            
    return probs

def probability_classes(dataset,s):
    y_dict = collections.Counter(y)
    n_classes = len(y_dict)
    n_cols = dataset.shape[1]-1
    
    p_classes = classes_probability(y)
    m,v = mean_var(dataset)
    p= probability_features_classes(m,v,s)
    probs= np.ones(n_classes)
    
    for i in range(n_classes):
        produit=1
        for j in range(n_cols):
            produit*=p[i][j]
        produit*=p_classes[i]
        probs[i]=produit
    return probs


    
s= [6,130,8]
probs=probability_classes(dataset,s)


