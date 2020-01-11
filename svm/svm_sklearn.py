#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:09:16 2019

@author: bahalla
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

raw_data = pd.read_csv("/home/bahalla/Documents/S5/naji/ML/datasets/mushrooms.csv")

labelEncoder=LabelEncoder()

for col in raw_data.columns:
    raw_data[col] = labelEncoder.fit_transform(raw_data[col])


x = raw_data.iloc[:,1:23]
y = raw_data.iloc[:,0]

x_train, y_train, x_test, y_test = train_test_split(x,y, test_size=0.33)


clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test,y_pred))