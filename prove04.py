#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:57:13 2020
@author: marcus
"""
#INSTALL WITH "PIP3 INSTALL <library>"
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from statistics import median
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log

dataset = {'Taste':['Salty','Spicy','Spicy','Spicy','Spicy','Sweet','Salty','Sweet','Spicy','Salty'],
       'Temperature':['Hot','Hot','Hot','Cold','Hot','Cold','Cold','Hot','Cold','Hot'],
       'Texture':['Soft','Soft','Hard','Hard','Hard','Soft','Soft','Soft','Soft','Hard'],
       'Eat':['No','No','Yes','No','Yes','Yes','No','Yes','Yes','Yes']}

df = pd.DataFrame(dataset,columns=['Taste','Temperature','Texture','Eat'])

iris = datasets.load_iris()

names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, train_size=0.7, shuffle = True)

test = pd.DataFrame(X_test,columns=iris.feature_names)
df1 = test.sort_values(['sepal length (cm)'], ascending=[True])
a, b, c = np.split(df1, [int(.2*len(df1)), int(.5*len(df1))])

median([df1["sepal length (cm)"].iloc[0], df1["sepal length (cm)"].iloc[len(df1.index) - 1]])

#this will tell us which has the loset entropy so we know which column should go first
def calc_entropy(p):
    if p.empty != True:
        result = 0
        #print(len(p.index))
        for i in p:
            result += (-i * np.log2(i))
        return result
    else:
        return result
#print(calc_entropy(df1["sepal length (cm)"]))
#print(calc_entropy(df1["sepal width (cm)"]))
#print(calc_entropy(df1["petal length (cm)"]))
#print(calc_entropy(df1["petal width (cm)"]))

class Node:
    def __def__(self, attribute_name, leaf_value, node = None):
        self.attribute_name = attribute_name#column
        self.is_leaf = isnull(leaf)#is it a leaf
        self.leaf_value = leaf_value#value of the leaf
        self.children = dict()
        if node != None:
            for child in node:
                self.children[child.attribute_name] = child

class ID3:
    def __init__(self, height = 0):
        self.height = height
    
    def fit(self, train_data, target_data, feature_names):
        #self.data = train_data
        self.targets = target_data
        self.feature_names = feature_names
        self.data = pd.DataFrame(train_data,columns=iris.feature_names)
        self.tree = self.make_tree(self.data, self.feature_names, self.targets)
        return self.tree
    
    def split(data):
        if sizeof(data) % 2 == 0:
            return
        
    def make_tree(self, data, available_columns, targets):
        base_node = Node()
        #if all rows in data have same target
        for ele in data:
            #print(not any(data[ele].duplicated(keep=False)))
            if (not any(data[ele].duplicated(keep=False))):
                return base_node
        #if not more available comlumns
        if not available_columns:
            return base_node
        #if not more data
        if data.empty:
            return base_node
        
        vals = dict()
        for column in available_columns:
            #calc entropy. Molest columns goes first.
            vals[column] = calc_entropy(data[column])
        colums = sorted(vals.items(), key=lambda kv: kv[1])
        
        for col in reversed(colums):
            #make child node by calling make_tree
            cur = self
        for x in xrange(n):
            cur = cur.findSuccessor(cars)
            if cur is None:
                return None
        return cur
            #child_node = self.make_tree(data[col], col, self.targets)
            #add child to base_node
            base_node.children.append(child_node)
            #print(col)
    
    def predict(self, data_test):
        return True


id3 = ID3(5)
results = id3.fit(X_train, y_train, iris.feature_names)
print(results)
#targets_predicted = id3.predict(X_test)
#print(targets_predicted)

"""
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(predictions)
"""












