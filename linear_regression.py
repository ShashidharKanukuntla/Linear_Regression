#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:38:15 2018

@author: shashidhar

Linear Regression Algorithm by Gradient Descent Optimization without using sklearn
"""

# Importing Packages
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Function to Randomly Initialize weights 
def initializeweights(traindata):
    return np.random.randn(traindata.shape[1],1)

# Function for Splitting the dataset to training and test sets according to split ratio
def traintestsplit(data,labels,splitratio):
    split = math.ceil(splitratio*X.shape[0])
    x_tr = X[:-split,:]
    x_te = X[-split:,:]
    y_tr = y[:-split]
    y_te = y[-split:]
    return x_tr, x_te, y_tr, y_te

# Fuction to Normalize data
def normalizedata(data):
    for i in range(data.shape[1]):
        data[:,i]=data[:,i]/(np.max(data[:,i]))
    return data
    
# Calculating Mean square error
def meansquareerror(pred,actual):
    return np.sum((pred-actual)*(pred-actual))/(2*actual.shape[0])

# Function to Add bias term to data 
def addbias(data):
    tmp = [1]*data.shape[0]
    tmp = np.array(tmp).reshape(data.shape[0],1)
    return np.concatenate((tmp,data),axis =1)

# Predicting the output by using trained model
def prediction(data,weights):
    return np.dot(data,weights)

# Fitting the model on training set using gradient descent optimization
def fitmodel(data,labels,iterations,learning_rate):
    iter = 0
    k = np.random.randn(data.shape[1],1)
    h = prediction(data,k)
    while iter<iterations:
        delta = np.sum(data*(h-labels),axis =0)/(data.shape[0])
        delta = delta.reshape(data.shape[1],1)
        k = k-(learning_rate*delta)
        h = prediction(data,k)
        iter = iter +1
    return k
    
# Importing data set
dataset = pd.read_csv('Salary_Data.csv')

# Splitting the dataset into features and labels
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into training and test sets
x_train, x_test, y_train, y_test = traintestsplit(X,y,0.15)

# Normalizing training and test sets
#x_train = normalizedata(x_train)
x_train = x_train/(np.max(x_train))
x_test = x_test/(np.max(x_test))
#x_test = normalizedata(x_test)

# adding bias term to training and test sets
newtrain = addbias(x_train)
newtest = addbias(x_test)

# Avoiding numpy array broadcast problem by reshaping accordingly
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

# Fitting the model on training set
model = fitmodel(newtrain,y_train,5000,0.1)

# Predicting the model on test set
pred = prediction(newtest,model)

print('original-',y_test) # Printing test set labels
print('predicted-',pred) # Printing test set prediction labels

plt.plot(y_test, color = 'red', label = 'Real salary')
plt.plot(pred, color = 'blue', label = 'Predicted salary')
plt.title('salary prediction')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
# calculating the mean sqaure error and printing
cost = meansquareerror(pred,y_test)
print('mean_square_error-',cost)
