#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:44:48 2019

@author: bahalla
"""

# Packages for analysis
import pandas as pd
import numpy as np
from sklearn import svm
# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
 
recipes = pd.read_csv('/home/bahalla/Documents/S5/naji/ML/datasets/recipes_muffins.csv')

# Plot two ingredients
#sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type',
 #          palette='Set1', fit_reg=False, scatter_kws={"s": 70})

ingredients = recipes[['Flour','Sugar']].as_matrix()
type_label = np.where(recipes['Type']=='Muffin', 0, 1)
# Feature names
recipe_features = recipes.columns.values[1:].tolist()

#fit the model and use svm algorithm
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)


# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

#plot the results
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black');
#plot the hyperplane
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=80, facecolors='none')

#function to predict if it is muffin or cupcake
def predict_muffin_or_cupcake(flour, sugar):
    if(model.predict([[flour, sugar]]))==0:
        print('c\'est du muffin')
    else:
        print('c\'est du capcake')

predict_muffin_or_cupcake(37, 20)
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(50, 20, 'yo', markersize='9');
