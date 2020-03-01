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
import pydotplus
from sklearn import tree
import collections
import seaborn as sns
#%matplotlib inline

# importing the required module 
import matplotlib.pyplot as plt 
import seaborn as sns

eps = np.finfo(float).eps
from numpy import log2 as log
#Import the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

""""
#Import the dataset 
dataset = pd.read_csv('data/zoo.csv')
#We drop the animal names since this is not a good feature to split the data on
dataset=dataset.drop('animal_name',axis=1)
"""

iris = datasets.load_iris()
#iris = pd.Series(iris)
#print(iris.isna().any()) # shows which columns have NA's


names = iris.target_names
train_features, test_features, train_targets, test_targets = train_test_split(iris.data, iris.target, test_size=0.3, train_size=0.7, shuffle = True)

target_names = iris.target_names
data_feature_names = iris.feature_names

tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)

# Training
"""
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_features,train_targets)

# Visualize data
dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')
"""
#########################################################################

prediction = tree.predict(test_features)


print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")


# x axis values 
x =['sun', 'mon', 'fri', 'sat', 'tue', 'wed', 'thu'] 
  
# y axis values 
y =[5, 6.7, 4, 6, 2, 4.9, 1.8] 
# plotting strip plot with seaborn 
ax = sns.stripplot(x, y); 
  
# giving labels to x-axis and y-axis 
ax.set(xlabel ='x', ylabel ='y') 
  
# giving title to the plot 
plt.title('My first graph'); 
  
# function to show plot 
plt.show() 

"""
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(predictions)
"""












