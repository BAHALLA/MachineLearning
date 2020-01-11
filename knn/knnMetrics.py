#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 00:25:17 2020

@author: bahalla
"""

import pandas as pn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

dataset = pn.read_csv("/home/bahalla/Documents/S5/naji/ML/datasets/diabetes.csv")

outcome = dataset.iloc[:,8]

dataset_train,dataset_test,outcome_train,outcome_test= train_test_split(dataset,outcome,random_state=1,test_size=0.2)

classifier=KNeighborsClassifier(n_neighbors=3,p=2,metric='euclidean')
classifier.fit(dataset_train,outcome_train)
outcome_pred=classifier.predict(dataset_test)
#confusion matrix
cm= confusion_matrix(outcome_test,outcome_pred)
#accuracy
ascore = accuracy_score(outcome_test,outcome_pred)
#f1_score
f1=f1_score(outcome_test,outcome_pred)
#precision
precision = precision_score(outcome_test,outcome_pred)

print(ascore)
print(precision)

