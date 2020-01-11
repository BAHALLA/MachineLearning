#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:13:24 2020
@author: bahalla
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

import pandas as pd



iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

#split data to train data and test data 
x_train,x_test, y_train,y_test = train_test_split(df[iris.feature_names], df['target'], random_state=0 )

#decision tree classifier instance
classifier = DecisionTreeClassifier(max_depth=3, random_state=0)
#train the modele
classifier.fit(x_train, y_train)

#prediction
prd = classifier.predict(x_test.iloc[3].values.reshape(1, -1))
#print(prd)
#multiple predictions
mlpPrd = classifier.predict(x_test[0:10])

#metrics
score = classifier.score(x_test, y_test)
print(score)

#
n_leaves = classifier.get_n_leaves()
print(n_leaves)

