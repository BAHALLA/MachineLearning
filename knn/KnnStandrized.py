#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 01:15:31 2020

@author: bahalla
"""


import numpy as pn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv("/home/bahalla/Documents/S5/naji/ML/datasets/diabetes.csv")

zero_not_accepted=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0,pn.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(pn.NaN, mean)
    
x=dataset.iloc[:,0:8]
y=dataset.iloc[:,8]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


#standarized dataset
#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#x_test =  scaler.transform(x_test)

#normlized dataset
normlize = MinMaxScaler()
x_train = normlize.fit_transform(x_train)
x_test = normlize.transform(x_test)
#Knn classifier
classifier=KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print(accuracy_score(y_test, y_pred))    
print(precision_score(y_test, y_pred))