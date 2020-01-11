#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 22:10:40 2020

@author: bahalla
"""

from collections import Counter
import math

def knn(dataset, query , k , distance_function, choice_function):
    
    #collection for distance between searched class element and dataset and indeces
    neighbor_distance_indices = []
    
    #for each element in data and query calculate and store distance in collection 
    for index, element in enumerate(dataset):
        distance = distance_function(element[:-1], query)
        
        neighbor_distance_indices.append((distance, index))
        
    
    #sort distances
    sorted_neighbor_distance_indices = sorted(neighbor_distance_indices)
  
    #get k nearest distances
    k_nearest_distance_indices = sorted_neighbor_distance_indices[:k]
    
    #get labels of k nearest distances
    k_labels = [dataset[i][1] for distance, i in k_nearest_distance_indices ]
    
    #return k distances and result (classification)
    return k_nearest_distance_indices, k_labels, choice_function(k_labels)

#classification choice function
def mode(labels):
    return Counter(labels).most_common(1)[0][0]

#euclidean distance 
def euclidean_distance(point1, point2):
    sum_distance = 0
    
    for i in range(len(point1)):
        sum_distance += math.pow((point1[i] - point2[i]),2)
    return math.sqrt(sum_distance)

def main():
    data = [[22, 1],[23, 1],[21, 1],[18, 1],[19, 1],[25, 0],[27, 0],[29, 0],[31, 0],[45, 0]] 
    query =[24]
    k_nearest,k_labels, class_prediction = knn(data, query, 3, distance_function = euclidean_distance,choice_function = mode)
   
    print(class_prediction)
    print("-------------------------------------------")
    print(k_nearest)
    print("-------------------------------------------")
    print(k_labels)
  

if __name__ == '__main__':
    
    main()

    
        
    
        