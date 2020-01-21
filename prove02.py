#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:57:13 2020

@author: marcus
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, train_size=0.7, shuffle = True)

class kNN:
    def __init__(self, k = 0):
        self.k = k
    
    def fit(self, train_data, targets):
        self.data = train_data
        self.targets = targets
        return
    
    def predict(self, data_test):
        list = []
        for data in data_test:
            for element in self.data:
                list.append(predict_single(data, element))
                
            list.sort()
            return list[slice(self.k)]
        
        return list

def predict_single(test_row, test_data):
    #kNN
    dist = []
    for i in range(0, 4):
        dist.append(np.sum((test_row[i] - test_data[i]) ** 2))
        
    return np.sqrt(dist[0] + dist[1] + dist[2] + dist[3])


machine = kNN(3)
machine.fit(X_train, y_train)

targets_predicted = machine.predict(X_test)

print(targets_predicted)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(predictions)













