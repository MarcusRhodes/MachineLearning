import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsRegressor

# ...
# ... code here to load a training and testing set
car_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

# Load the car data
car_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", header=None, names=car_names, skipinitialspace=True, na_values=["?"])

car_data.head()

# See if we have any NA's right now
car_data[car_data.isna().any(axis=1)] # shows records with NA's
car_data.isna().any() # shows which columns have NA's

# See if we have any NA's right now (This should not show us any now...)
car_data[car_data.isna().any(axis=1)] # shows records with NA's
car_data.isna().any() # shows which columns have NA's

# buying
car_data.buying.value_counts()
car_data.buying = car_data.buying.astype('category')
car_data["buying_cat"] = car_data.buying.cat.codes

# maint
car_data.maint.value_counts()
car_data.maint = car_data.maint.astype('category')
car_data["maint_cat"]= car_data.maint.cat.codes

# doors
car_data.doors.value_counts()
car_data.doors = car_data.doors.astype('category')
car_data["doors_cat"]= car_data.doors.cat.codes

# persons
car_data.persons.value_counts()
car_data.persons = car_data.persons.astype('category')
car_data["persons_cat"]= car_data.persons.cat.codes

# lug_boot
car_data.lug_boot.value_counts()
car_data.lug_boot = car_data.lug_boot.astype('category')
car_data["lug_boot_cat"]= car_data.lug_boot.cat.codes

# safety
car_data.safety.value_counts()
car_data.safety = car_data.safety.astype('category')
car_data["safety_cat"]= car_data.safety.cat.codes

# Finally, let's get rid of all of the old columns
car_data = car_data.drop(columns=["buying", "maint", "doors", "persons", "lug_boot", "safety"])
#print(car_data.safety_cat)
X_train, X_test, y_train, y_test = train_test_split(car_data, car_data.safety_cat, test_size=0.3, train_size=0.7, shuffle = True)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
#print(predictions)



#MPG DATA SET!!!
mpg_names = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car_name"]

# Load the mpg data
mpg_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", header=None, names=mpg_names, delim_whitespace=True, skipinitialspace=True, na_values=["?"])
mpg_data.head()

# See if we have any NA's right now
#print(mpg_data[mpg_data.isna().any(axis=1)]) # shows records with NA's
#print(mpg_data.isna().any()) # shows which columns have NA's

#mpg_data.horsepower = mpg_data.horsepower.fillna("unknown")
mpg_data.horsepower = mpg_data.horsepower.fillna("0")

# car_name
mpg_data.car_name.value_counts()
mpg_data.car_name = mpg_data.car_name.astype('category')
mpg_data["car_name_cat"]= mpg_data.car_name.cat.codes
mpg_data = mpg_data.drop(columns=["car_name"])

X_train, X_test, y_train, y_test = train_test_split(mpg_data, mpg_data.mpg, test_size=0.3, train_size=0.7, shuffle = True)
regr = KNeighborsRegressor(n_neighbors=3)
regr.fit(X_train, y_train)
predictions = regr.predict(X_test)
#print(predictions)







#STUDENT DATA SET!!!
student_names = ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","guardian","traveltime","studytime","failures","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic","famrel","freetime","goout","Dalc","Walc","health","absences"]

# Load the student data
student_data = pd.read_csv("data/student-mat.csv", sep = ';')
student_data.head()

# See if we have any NA's right now
student_data[student_data.isna().any(axis=1)] # shows records with NA's
student_data.isna().any()# shows which columns have NA's

# See if we have any NA's right now (This should not show us any now...)
student_data[student_data.isna().any(axis=1)] # shows records with NA's
student_data.isna().any() # shows which columns have NA's

# school
student_data.school.value_counts()
student_data.school = student_data.school.astype('category')
student_data["school_cat"] = student_data.school.cat.codes

# sex
student_data.sex.value_counts()
student_data["sex"] = student_data.sex.map({"Male": 1, "Female": 0})

# address
student_data.address.value_counts()
student_data.address = student_data.address.astype('category')
student_data["address_cat"]= student_data.address.cat.codes

# famsize
student_data.famsize.value_counts()
student_data.famsize = student_data.famsize.astype('category')
student_data["famsize_cat"]= student_data.famsize.cat.codes

# Pstatus
student_data.Pstatus.value_counts()
student_data.Pstatus = student_data.Pstatus.astype('category')
student_data["Pstatus_cat"]= student_data.Pstatus.cat.codes

# Mjob
student_data.Mjob.value_counts()
student_data.Mjob = student_data.Mjob.astype('category')
student_data["Mjob_cat"]= student_data.Mjob.cat.codes

# Fjob
student_data.Fjob.value_counts()
student_data.Fjob = student_data.Fjob.astype('category')
student_data["Fjob_cat"]= student_data.Fjob.cat.codes

# reason
student_data.reason.value_counts()
student_data.reason = student_data.reason.astype('category')
student_data["reason_cat"]= student_data.reason.cat.codes

# guardian
student_data.guardian.value_counts()
student_data.guardian = student_data.guardian.astype('category')
student_data["guardian_cat"]= student_data.guardian.cat.codes

# schoolsup
student_data.schoolsup.value_counts()
student_data.schoolsup = student_data.schoolsup.astype('category')
student_data["schoolsup_cat"]= student_data.schoolsup.cat.codes

# famsup
student_data.famsup.value_counts()
student_data.famsup = student_data.famsup.astype('category')
student_data["famsup_cat"]= student_data.famsup.cat.codes

# paid
student_data.paid.value_counts()
student_data.paid = student_data.paid.astype('category')
student_data["paid_cat"]= student_data.paid.cat.codes

# activities
student_data.activities.value_counts()
student_data.activities = student_data.activities.astype('category')
student_data["activities_cat"]= student_data.activities.cat.codes

# nursery
student_data.nursery.value_counts()
student_data.nursery = student_data.nursery.astype('category')
student_data["nursery_cat"]= student_data.nursery.cat.codes

# higher
student_data.higher.value_counts()
student_data.higher = student_data.higher.astype('category')
student_data["higher_cat"]= student_data.higher.cat.codes

# internet
student_data.internet.value_counts()
student_data.internet = student_data.internet.astype('category')
student_data["internet_cat"]= student_data.internet.cat.codes

# romantic
student_data.romantic.value_counts()
student_data.romantic = student_data.romantic.astype('category')
student_data["romantic_cat"]= student_data.romantic.cat.codes

# Finally, let's get rid of all of the old columns
student_data = student_data.drop(columns=["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "internet", "romantic", "higher"])

X_train, X_test, y_train, y_test = train_test_split(student_data, student_data.G3, test_size=0.3, train_size=0.7, shuffle = True)
regr = KNeighborsRegressor(n_neighbors=3)
regr.fit(X_train, y_train)
predictions = regr.predict(X_test)
print(predictions)
print(y_test)








