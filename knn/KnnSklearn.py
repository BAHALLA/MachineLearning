#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 23:10:12 2020

@author: bahalla
"""

from sklearn.neighbors import KNeighborsClassifier

data = [[22],[23],[21],[18],[19],[25],[27],[29],[31],[45]] 
label = [1,1,1,1,1,0,0,0,0,0]

k = 3

query = [[33]]

classifie = KNeighborsClassifier(n_neighbors = k)

classifie.fit(data, label)

prediction = classifie.predict(query)

print(prediction)