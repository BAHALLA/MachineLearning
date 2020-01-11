#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 21:17:41 2019

@author: bahalla
"""


import numpy as np
import collections
import pandas as pn
import math
from statistics import variance

dataset = pn.DataFrame({"Flower": ["Iris-setosa","Iris-setosa","Iris-setosa",
                                   "Iris-versicolor","Iris-versicolor","Iris-versicolor",
                                   "Iris-virginica","Iris-virginica","Iris-virginica"],
                        "sepal_length":[5.1,4.9,4.7,7.0,6.4,6.9,6.4,6.5,7.7],
                        "sepal_width":[3.5,3.0,3.2,3.2,3.2,3.1,3.2,3.0,3.8],
                        "petal_length":[1.4,1.4,1.3,4.7,4.5,4.9,5.3,5.5,6.7],
                        "petal_width":[0.2,0.2,0.3,1.4,1.5,1.5,2.3,1.8,2.2]
                        })


def pdf(x,m,v):
    return 1/math.sqrt(2*math.pi*v)*math.exp(-0.5*math.pow(x-m,2)/v)

def split(dataset):
    x=dataset.iloc[:,1:]
    y=dataset.loc[:,'Flower']

    return x,y

x,y = split(dataset)
def classes_probability(y):
    y_dict = collections.Counter(y)
    n_classes = len(y_dict)
    #get keys of classes
    keys_cls = list(y_dict.keys())
    prob=np.ones(n_classes)
    nb_elem = y.shape[0]
    for i in range(n_classes):
         prob[i] = y_dict[keys_cls[i]]/nb_elem
         print( keys_cls[i])
     
    return  prob

def mean_var(dataset):
#m = moyenne and v = variance
    y_dict = collections.Counter(y)
    n_classes = len(y_dict)
    n_cols = dataset.shape[1]-1
    m= np.ones((n_classes,n_cols))
    v= np.ones((n_classes,n_cols))
    
    ise = dataset.loc[dataset['Flower'] == "Iris-setosa",:]
    ive = dataset.loc[dataset['Flower'] == "Iris-versicolor",:]
    ivi = dataset.loc[dataset['Flower'] == "Iris-virginica",:]
    
    for j in range(n_cols):
            col = ise.iloc[:,j+1]
            m[0][j] = np.mean(col)
            v[0][j] = variance(col)
    for j in range(n_cols):
            col = ive.iloc[:,j+1]
            m[1][j] = np.mean(col)
            v[1][j] = variance(col)
    for j in range(n_cols):
            col = ivi.iloc[:,j+1]
            m[2][j] = np.mean(col)
            v[2][j] = variance(col)
    
    return m,v

m,v = mean_var(dataset)

def probability_features_classes(m,v,test):
    n_classes =m.shape[0]
    n_cols = m.shape[1]
    probs=np.ones((n_classes,n_cols))
    
    for i in range(n_classes):
        for j in range(n_cols):
            probs[i][j] = pdf(test[j], m[i][j],v[i][j])
            
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

s =[4.9,3.0,4.9,2.1]
probs = probability_classes(dataset,s)








