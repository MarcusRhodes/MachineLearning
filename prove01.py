#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:57:13 2020

@author: marcus
"""
#import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
#print(iris.data)

# Show the target values (in numeric format) of each instance
#print(iris.target)

# Show the actual target names that correspond to each number
#print(iris.target_names)

iris_train, iris_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.3, train_size=0.7, shuffle = True)

classifier = GaussianNB()
classifier.fit(iris_train, target_train)

targets_predicted = classifier.predict(iris_test)
#print(targets_predicted)
#print(iris.target)

class HardCodedClassifier:
    
    def fit(self, iris_train, target_train):
        return True
    
    def predict(self, data_test):
        # print(error)
        list = []
        for data in data_test:
            list.append(data)
        return list


classifier = HardCodedClassifier()
classifier.fit(iris_train, target_train)
# print(iris_train)
targets_predicted = classifier.predict(iris_test)
print(targets_predicted)


