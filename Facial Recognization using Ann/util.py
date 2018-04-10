# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 12:43:58 2018

@author: rishabh
"""

import numpy as np

def sigmoid(A):
  return 1/(1+np.exp(-A))
  
def sigmoidCost(T,Y):
  return -(T*np.log(Y)+(1-T)*np.log(1-Y)).sum()

def cost(T,Y):
  return -(T*np.log(Y)).sum()
  
def relu(x):
  return x*(x>0)

def errorRate(target,predictions):
  return np.mean(target!=predictions)

def softmax(A):
  expA=np.exp(A)
  return expA/expA.sum(axis=1,keepdims=True)

def y2indicator(Y):
  N=len(Y)
  K=len(set(Y))
  T=np.zeros((N,K))
  for i in range(N):
    T[i,Y[i]]=1
  return T
    
def getData(balance_ones=True):
  Y=[]
  X=[]
  
  first=True
  for line in open('E:/RS/ML/Machine learning tuts/Target/Projects/fer2013/fer2013.csv'):
    if first:
      first=False
    else:
      row=line.split(',')
      y=int(row[0])
      Y.append(y)
      X.append([int(p) for p in row[1].split()])
          
  X,Y = np.array(X)/255,np.array(Y)  

  if balance_ones==True:
    X0,Y0=X[Y!=1,:],Y[Y!=1]
    X1=X[Y==1,:]
    X1=np.repeat(X1,9,axis=0)
    X=np.vstack([X0,X1])
    Y=np.concatenate((Y0,[1]*len(X1)))
    
  return X,Y
    
def getBinaryData():
  Y=[]
  X=[]
  first=True
                
  for line in open('E:/RS/ML/Machine learning tuts/Target/Projects/fer2013/fer2013.csv'):
    if first:
      first=False
    else:
      row=line.split(',')
      y=int(row[0])
      if y==0 or y==1:
        Y.append(y)
        X.append([int(p) for p in row[1].split()])
  return np.array(X)/255,np.array(Y)
  
