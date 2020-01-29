"""import pandas as pd
import numpy as np
"""
"""
names = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","race","sex","capital-loss","hours-per-week","native-country","target"]

# Read in the CSV file and convert "?" to NaN
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", names=names, skipinitialspace=True, na_values=["?"])

#label encoding
data_new = pd.get_dummies(data, columns=["occupation"])
data["occupation"] = data.occupation.astype("category")#data.occupation.cat.codes

print(data_new.head())"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS 450 - 03 Teach : Team Activity
This script shows one approach to the team activity. It is not the
only way, and there may be improvements to various things throughout.
Data set source: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
@author: Brother Burton
"""

import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##############################
# Part 1 - Read in the data
##############################

# Make sure the script is running in the directory I expect
#os.chdir("/Users/sburton/git/byui-cs/cs450-faculty/teacher-solutions")

names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
         "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
         "hours_per_week", "native_country", "income"]

# Load the file
#data = pd.read_csv("data/adult_data.txt", header=None, skipinitialspace=True, names=names, na_values=["?"])
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", names=names, skipinitialspace=True, na_values=["?"])

# Print some summaries of the data for sanity sake
print(data)
print(data.columns)
print(data.dtypes)

print(data.age.median())
print(data.native_country.value_counts())

##############################
# Part 2 - Handle missing data
##############################

# See if we have any NA's right now
data[data.isna().any(axis=1)] # shows records with NA's
data.isna().any() # shows which columns have NA's

# In my case, I am going to set the missing data to a new
# category called, "unknown". Other good options would include:
# -- Using the most common value for that attribute
# -- Dropping the row completely if it has missing data
# -- Imputing (or filling in) the value with a more sophisticated way
data.workclass = data.workclass.fillna("unknown")
data.native_country = data.native_country.fillna("unknown")
data.occupation = data.occupation.fillna("unknown")

# See if we have any NA's right now (This should not show us any now...)
data[data.isna().any(axis=1)] # shows records with NA's
data.isna().any() # shows which columns have NA's


##############################
# Part 3 - Convert to Numeric
##############################

# Following the ideas from: http://pbpython.com/categorical-encoding.html

# Show a list of the columns that are "object" types
print(data.select_dtypes(include=["object"]).columns)

# Two main choices here, one-hot enconding or label encoding, for the
# most part here, I opted for label encoding, but I used one-hot encoding
# for the race


# For each one, I did a value_counts() first to see what it looked like,
# if it's a really big list of possibilities, I consider label encoding

# Workclass
data.workclass.value_counts()
data.workclass = data.workclass.astype('category')
data["workclass_cat"] = data.workclass.cat.codes

# education
data.education.value_counts()
data.education = data.education.astype('category')
data["education_cat"]= data.education.cat.codes

# marital_status
data.marital_status.value_counts()
data.marital_status = data.marital_status.astype('category')
data["marital_status_cat"]= data.marital_status.cat.codes

# occupation
data.occupation.value_counts()
data.occupation = data.occupation.astype('category')
data["occupation_cat"]= data.occupation.cat.codes

# relationship
data.relationship.value_counts()
data.relationship = data.relationship.astype('category')
data["relationship_cat"]= data.relationship.cat.codes

# race
data.race.value_counts()
#####
# Using one hot encoding for this one...
#####
data = pd.get_dummies(data, columns=["race"])

# sex
data.sex.value_counts()
data["isMale"] = data.sex.map({"Male": 1, "Female": 0})

# native_country
data.native_country.value_counts()
data.native_country = data.native_country.astype('category')
data["native_country_cat"]= data.native_country.cat.codes

# income (our target)
data.income.value_counts()
data["incomeHigh"] = data.income.map({">50K": 1, "<=50K": 0})


# Finally, let's get rid of all of the old columns
# NOTE: Race has already been dropped
data = data.drop(columns=["workclass", "education", "marital_status", "occupation",
                   "relationship", "sex", "native_country", "income"])


####################################
# Part 4 - Use sk-learn to predict
####################################
# First convert the data to numpy arrays, because that's what sk-learn likes
X = data.drop(columns=["incomeHigh"]).as_matrix()
y = data["incomeHigh"].as_matrix().flatten()

# Break up into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the classifier
classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Compute and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))

# Stretch challenges still to do...
# Normalize the numeric attributes
# Use k-fold cross validation