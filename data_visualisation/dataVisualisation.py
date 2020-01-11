#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:45:40 2019

@author: bahalla
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pn

dataset=pn.read_csv("/home/bahalla/Documents/S5/naji/ML/datasets/iris_dataset.csv")

C1=dataset.loc[dataset['flower']=="Iris-setosa",:]
C0=dataset.loc[dataset['flower']=="Iris-versicolor",:]
C2=dataset.loc[dataset['flower']=="Iris-virginica",:]
x1=C1.loc[:,'sepal_length']
y1=C1.loc[:,'sepal_width']

x2=C0.loc[:,'sepal_length']
y2=C0.loc[:,'sepal_width']

x3=C2.loc[:,'sepal_length']
y3=C2.loc[:,'sepal_width']
plt.plot(x1, y1,'g.')
plt.plot(x2, y2,'r.')
plt.plot(x3, y3,'b.')