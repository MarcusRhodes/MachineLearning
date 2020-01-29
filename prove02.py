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
from statistics import mode

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
        #print(len(self.targets))
        list = []
        list2 = []
        distance = []
        for data in data_test:
            #print(data)
            for element in self.data:
                #print(element)
                #return np.sqrt(np.sum((data - element) ** 2))
                value = np.sqrt(np.sum((data - element) ** 2))
                distance.append( value )
            sorted = np.argsort(distance)
            sorted = sorted[:self.k]
            for s in sorted:
                if s <= len(self.targets):
                    list2.append(self.targets[s])
            if len(list2) != 0: 
                list.append(mode(list2))
            #print(list2)
            del list2[:]
            
        return list

def predict_single(test_row, test_data):
    #kNN
    return np.sqrt(np.sum((test_row - test_data) ** 2))

machine = kNN(3)
machine.fit(X_train, y_train)
targets_predicted = machine.predict(X_test)
print(targets_predicted)

"""
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(predictions)
"""












