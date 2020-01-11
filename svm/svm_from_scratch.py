#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:26:59 2019

@author: bahalla
"""
import pandas as pd
import numpy as np
 
recipes = pd.read_csv('/home/bahalla/Documents/S5/naji/ML/datasets/recipes_muffins.csv')
ingredients = recipes[['Flour','Sugar']].as_matrix()
type_label = np.where(recipes['Type']=='Muffin', -1, 1)

#retourner nombre d'attributs n et le nombre exemple de training
def paramsize(X):    
    m=X.shape[0]
    n=X.shape[1]
    return n,m

#initializer n par des zeros
def initParam(n):
    return np.zeros((n,1))
#calculer w.x+b
def compute(X,weights,b):   
    return  np.dot(X,weights) + b

#calculer d'erreur 
def compute_cost(Y,pred): 
    value=Y*pred
    if value >1:
        cost=0
    else :
        cost=1-value
    return cost
#train de modele pour trouver ||w|| minimum
def svm(X,Y,epochs=1000,lr=1):    
    n,m=paramsize(X)
    w=initParam(n)       
    for iter in range(1,epochs):
        for i,x in enumerate(X):
            x_train=x.reshape(1,-1)    
            pred=compute(x_train,w,1/iter) 
            cost=compute_cost(Y[i].reshape(1,1),pred)
            if cost==0:
                grad=np.zeros(w.shape)
            else:
                grad=(-Y[i].reshape(1,1)*x_train).T
            grad=grad+2/iter*w
            w=w-lr*grad
    return w

#pridection function
def predict(features,w,b):
        classification = np.sign(np.dot(np.array(features),w)+b)
        if classification >= 0:
            print('capcake')
        else:
            print('muffin')
       
w = svm(ingredients,type_label)
pred = compute([10,20],w,0.01)
