#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:37:17 2019

@author: bahalla
"""
import pandas as pd
import numpy as np

recipes = pd.read_csv('/home/bahalla/Documents/S5/naji/ML/datasets/recipes_muffins.csv')
ingredients = recipes[['Flour','Sugar']].as_matrix()
type_label = np.where(recipes['Type']=='Muffin', -1, 1)
 
def fit(data):
        
        opt_dict = {}
        
        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]
        max_feature_value = 55        
      
       
        #with smaller steps our margins and db will be more precise
        step_sizes = [max_feature_value * 0.1,
                      max_feature_value * 0.01,
                      #point of expense
                      max_feature_value * 0.001,]
        
        #extremly expensise
        b_range_multiple = 5
        #we dont need to take as small step as w
        b_multiple = 5
        
        latest_optimum = max_feature_value*10
   
        #making step smaller and smaller to get precise value
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            
            #we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*max_feature_value*b_range_multiple,
                                   max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        
                        #weakest link in SVM fundamentally
                        #SMO attempts to fix this a bit
                        # ti(xi.w+b) >=1
                        i=0
                        for xi in data:
                            yi=type_label[i]
                            if not yi*(np.dot(w_t,xi)+b)>=1:
                                found_option=False
                            i=i+1
                        if found_option:
                           opt_dict[np.linalg.norm(w_t)]=[w_t,b]
                if w[0]<0:
                    optimized=True
                    print("optimized a step")
                else:
                    w = w-step
                    
            # sorting ||w|| to put the smallest ||w|| at poition 0 
            norms = sorted([n for n in opt_dict])
            #optimal values of w,b
            opt_choice = opt_dict[norms[0]]

            w=opt_choice[0]
            b=opt_choice[1]
            
            latest_optimum = opt_choice[0][0]+step*2
        return w,b
w,b = fit(ingredients)
