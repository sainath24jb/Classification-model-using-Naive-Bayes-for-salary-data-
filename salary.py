# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 00:32:04 2019

@author: Hello
"""

import pandas as pd
import numpy as np
salary_train = pd.read_csv("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\Naive Bayes\\Datasets\\SalaryData_Train.csv")
salary_test = pd.read_csv("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\Naive Bayes\\Datasets\\SalaryData_Test.csv")
string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

##Preprocessing the data. As, there are categorical variables
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
for i in string_columns:
    salary_train[i]= number.fit_transform(salary_train[i])
    salary_test[i]=number.fit_transform(salary_test[i])
    

##Capturing the column names which can help in futher process
colnames = salary_train.columns
colnames
len(colnames)

x_train = salary_train[colnames[0:13]]
y_train = salary_train[colnames[13]]
x_test = salary_test[colnames[0:13]]
y_test = salary_test[colnames[13]]

##Building Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

##Building the Multinomial Naive Bayes Model
classifier_mb = MB()
classifier_mb.fit(x_train,y_train)
pred_mb = classifier_mb.predict(x_train)
accuracy_mb_train = np.mean(pred_mb == y_train)
##77%
pd.crosstab(pred_mb, y_train)

##for test data
pred_mb_test = classifier_mb.predict(x_test)
accuracy_mb_test = np.mean(pred_mb_test == y_test)
##77%
pd.crosstab(pred_mb_test, y_test)

##Building Gaussian model
classifier_gb = GB()
classifier_gb.fit(x_train, y_train)
pred_gb = classifier_gb.predict(x_train)
accuracy_gb_train = np.mean(pred_gb == y_train)
##80%
pd.crosstab(pred_gb,y_train)

##for test data
pred_gb_train = classifier_gb.predict(x_test)
accuracy_gb_test = np.mean(pred_gb_train == y_test)
##80%
pd.crosstab(pred_gb_train,y_test)
