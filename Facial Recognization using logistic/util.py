# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:43:20 2018

@author: rishabh
"""

import numpy as np
import pandas as pd 

def sigmoid(z):
  return 1/(1+np.exp(-z))

  
def sigmoid_cost(Yhat,Y):
  return -(Y*np.log(Yhat)+(1-Y)*np.log(1-Y)).sum()

  
def error_rate(targets,predictions):
  return np.mean(targets!=predictions)

def getBinaryData():
  Y=[]
  X=[]
  first=True
  for line in  open('E:/RS/ML/Machine learning tuts/Target/Part1(Regression)/Facial Recognization using logistic/fer2013/fer2013.csv'):
    if first:
      first=False
    else:
      row=line.split(',')
      y=int(row[0])
      if y==0 or y==1:
        Y.append(y)
        X.append([int(p) for p in row[1].split()])
  return np.array(X)/255.0,np.array(Y)