#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:00:42 2020

@author: marcus
"""
# importing the required module 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import keras as ks
import mnist

#ramen_names = ["Review_#", "Brand", "Variety", "Style", "Country", "Stars", "Top_Ten"]
# Load the ramen data
ramen_data = pd.read_csv("data/ramen-ratings.csv", sep = ',', header=0, skipinitialspace=True, na_values=["?"])
ramen_data.head()

#get rid of incomplete values
#ramen_data.Top_Ten = ramen_data.Top_Ten.fillna("0")
#or drop the column since it's trash
ramen_data = ramen_data.drop(columns=["Top Ten"])

#category encoding for string values
# Brand
ramen_data.Brand.value_counts()
ramen_data.Brand = ramen_data.Brand.astype('category')
ramen_data["brand_cat"]= ramen_data.Brand.cat.codes

# Variety
ramen_data.Variety.value_counts()
ramen_data.Variety = ramen_data.Variety.astype('category')
ramen_data["variety_cat"] = ramen_data.Variety.cat.codes

# Style
ramen_data.Style.value_counts()
ramen_data.Style = ramen_data.Style.astype('category')
ramen_data["style_cat"]= ramen_data.Style.cat.codes

# Country
ramen_data.Country.value_counts()
ramen_data.Country = ramen_data.Country.astype('category')
ramen_data["country_cat"]= ramen_data.Country.cat.codes

# Stars
#ramen_data.Stars.value_counts()
ramen_data.Stars = ramen_data.Stars.astype('category')
ramen_data["stars_cat"] = ramen_data.Stars.cat.codes

#sklearn standard scaler for normalizing

#Drop old columns
ramen_data = ramen_data.drop(columns=["Review #", "Brand", "Variety", "Style", "Country", "Stars"])
#ramen_data = ramen_data.drop(columns=["country_cat", "style_cat"])

#split for testing and training
x_train, x_test, y_train, y_test = train_test_split(ramen_data, ramen_data.stars_cat, test_size=0.3, train_size=0.7, shuffle = True)

#build Nural Network model
model = Sequential([
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(51, activation='softmax'),
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)
"""
model.fit(
  x_train.values, # training data
  ks.utils.to_categorical(y_train.values), # training targets
  epochs=7,#do it 7 times
  batch_size=32,#32 roes at a time
)
"""
history = model.fit(
  x_train.values, # training data
  ks.utils.to_categorical(y_train.values), # training targets
  epochs=15,#do it X times
  batch_size=32,#32 roes at a time
  validation_split = 0.2
)

import keras
from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model.evaluate(
  x_test,
  ks.utils.to_categorical(y_test)
)

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model
estimator = KerasRegressor(build_fn=history, epochs=10, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, x_train, y_train, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))



"""
toy_data = pd.read_csv("data/toy_dataset.csv", sep = ',', header=0, skipinitialspace=True, na_values=["?"])
toy_data.head()

#['Number', 'City', 'Gender', 'Age', 'Income', 'Illness']

#redundant column
toy_data = toy_data.drop(columns=["Number"])

#City
toy_data.City.value_counts()
toy_data.City = toy_data.City.astype('category')
toy_data["city_cat"]= toy_data.City.cat.codes

#Gender
toy_data.Gender.value_counts()
toy_data.Gender = toy_data.Gender.astype('category')
toy_data["gender_cat"]= toy_data.Gender.cat.codes

#Age
toy_data.Age.value_counts()
toy_data.Age = toy_data.Age.astype('category')
toy_data["age_cat"]= toy_data.Age.cat.codes

#Income
toy_data.Income.value_counts()
toy_data.Income = toy_data.Income.astype('category')
toy_data["income_cat"]= toy_data.Income.cat.codes

#Illness
toy_data.Illness.value_counts()
toy_data.Illness = toy_data.Illness.astype('category')
toy_data["illness_cat"]= toy_data.Illness.cat.codes

toy_data = toy_data.drop(columns=['City', 'Gender', 'Age', 'Income', 'Illness'])

#split for testing and training
x_train, x_test, y_train, y_test = train_test_split(toy_data, toy_data.illness_cat, test_size=0.3, train_size=0.7, shuffle = True)

#build Nural Network model
model = Sequential([
  Dense(64, activation='relu'),
  Dense(45, activation='relu'),
  Dense(32, activation='relu'),
  Dense(12, activation='relu'),
  #Dense(1, activation='sigmoid'),
  Dense(2, activation='softmax')
])

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.1), loss='mean_squared_error', metrics=['mae'])

history = model.fit(
  x_train.values, # training data
  ks.utils.to_categorical(y_train.values), # training targets
  epochs=15,#do it X times
  batch_size=32,#32 roes at a time
  validation_split = 0.2
)

import keras
from matplotlib import pyplot as plt
#plt.plot(history.history['accuracy'])
plt.plot(history.history['mae'])
#plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_mae'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()



from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, x_train, y_train, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))



#model.evaluate(
#  x_test,
#  ks.utils.to_categorical(y_test)
#)
#########################################################################

# importing the required module 
import seaborn as sns

prediction = model.evaluate(
  x_test,
  ks.utils.to_categorical(y_test)
)


#print("The prediction accuracy is: ",prediction*100,"%")


# x axis values 
x = x_test.columns
  
# y axis values 
y = y_test.values 
# plotting strip plot with seaborn 
ax = sns.stripplot(x, y); 
  
# giving labels to x-axis and y-axis 
ax.set(xlabel ='x', ylabel ='y') 
  
# giving title to the plot 
plt.title('My first graph'); 
  
# function to show plot 
plt.show() 

"""
from sklearn import datasets
#Iris cause I'm a chump ;)
iris = datasets.load_iris()

['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']
names = iris.target_names
"""





"""
#Video Games
vg_data = pd.read_csv("data/vgsales.csv", sep = ',', header=0, skipinitialspace=True, na_values=["?"])
vg_data.head()

#Rank
vg_data.Rank.value_counts()
vg_data.Rank = vg_data.Rank.astype('category')
vg_data["rank_cat"]= vg_data.Rank.cat.codes

#Name
vg_data.Name.value_counts()
vg_data.Name = vg_data.Name.astype('category')
vg_data["name_cat"]= vg_data.Name.cat.codes

#Rank
vg_data.Rank.value_counts()
vg_data.Rank = vg_data.Rank.astype('category')
vg_data["rank_cat"]= vg_data.Rank.cat.codes

#Platform
vg_data.Platform.value_counts()
vg_data.Platform = vg_data.Platform.astype('category')
vg_data["platform_cat"]= vg_data.Platform.cat.codes

#Year
vg_data.Year.value_counts()
vg_data.Year = vg_data.Year.astype('category')
vg_data["year_cat"]= vg_data.Year.cat.codes

#Genre
vg_data.Genre.value_counts()
vg_data.Genre = vg_data.Genre.astype('category')
vg_data["genre_cat"]= vg_data.Genre.cat.codes

#Publisher
vg_data.Publisher.value_counts()
vg_data.Publisher = vg_data.Publisher.astype('category')
vg_data["publisher_cat"]= vg_data.Publisher.cat.codes

#NA_Sales
vg_data.NA_Sales.value_counts()
vg_data.NA_Sales = vg_data.NA_Sales.astype('category')
vg_data["na_sales_cat"]= vg_data.NA_Sales.cat.codes

#EU_Sales
vg_data.EU_Sales.value_counts()
vg_data.EU_Sales = vg_data.EU_Sales.astype('category')
vg_data["eu_sales_cat"]= vg_data.EU_Sales.cat.codes

#JP_Sales
vg_data.JP_Sales.value_counts()
vg_data.JP_Sales = vg_data.JP_Sales.astype('category')
vg_data["jp_sales_cat"]= vg_data.JP_Sales.cat.codes

#Other_Sales
vg_data.Other_Sales.value_counts()
vg_data.Other_Sales = vg_data.Other_Sales.astype('category')
vg_data["other_sales_cat"]= vg_data.Other_Sales.cat.codes

#Global_Sales
vg_data.Global_Sales.value_counts()
vg_data.Global_Sales = vg_data.Global_Sales.astype('category')
vg_data["global_sales_cat"]= vg_data.Global_Sales.cat.codes

vg_data = vg_data.drop(columns=['Rank', 'Name', 'Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

x_train, x_test, y_train, y_test = train_test_split(vg_data, vg_data.rank_cat, test_size=0.3, train_size=0.7, shuffle = True)
#build Nural Network model
model = Sequential([
  Dense(124, activation='relu'),
  Dense(124, activation='relu'),
  Dense(124, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(1, activation='linear'),
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  x_train.values, # training data
  ks.utils.to_categorical(y_train.values), # training targets
  epochs=7,#do it 7 times
  batch_size=32,#32 roes at a time
)

model.evaluate(
  x_test,
  ks.utils.to_categorical(y_test)
)
"""








'''
#get rid of incomplete values
#ramen_data.Top_Ten = ramen_data.Top_Ten.fillna("0")
#or drop the column since it's trash
ramen_data = ramen_data.drop(columns=["Top Ten"])

#category encoding for string values
# Brand
ramen_data.Brand.value_counts()
ramen_data.Brand = ramen_data.Brand.astype('category')
ramen_data["brand_cat"]= ramen_data.Brand.cat.codes

# Variety
ramen_data.Variety.value_counts()
ramen_data.Variety = ramen_data.Variety.astype('category')
ramen_data["variety_cat"] = ramen_data.Variety.cat.codes

# Style
ramen_data.Style.value_counts()
ramen_data.Style = ramen_data.Style.astype('category')
ramen_data["style_cat"]= ramen_data.Style.cat.codes

# Country
ramen_data.Country.value_counts()
ramen_data.Country = ramen_data.Country.astype('category')
ramen_data["country_cat"]= ramen_data.Country.cat.codes

# Stars
#ramen_data.Stars.value_counts()
ramen_data.Stars = ramen_data.Stars.astype('category')
ramen_data["stars_cat"] = ramen_data.Stars.cat.codes

#sklearn standard scaler for normalizing

#Drop old columns
ramen_data = ramen_data.drop(columns=["Review #", "Brand", "Variety", "Style", "Country", "Stars"])
#ramen_data = ramen_data.drop(columns=["country_cat", "style_cat"])

#split for testing and training
x_train, x_test, y_train, y_test = train_test_split(ramen_data, ramen_data.stars_cat, test_size=0.3, train_size=0.7, shuffle = True)

#build Nural Network model
model = Sequential([
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(51, activation='softmax'),
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  x_train.values, # training data
  ks.utils.to_categorical(y_train.values), # training targets
  epochs=7,#do it 7 times
  batch_size=32,#32 roes at a time
)

model.evaluate(
  x_test,
  ks.utils.to_categorical(y_test)
)
'''





