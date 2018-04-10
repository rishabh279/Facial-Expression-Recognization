# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:57:47 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt

from util import getData,softmax,cost,y2indicator,errorRate
from sklearn.utils import shuffle

class LogisticModel(object):
  def __init__(self):
    pass
  
  def forward(self,X):
    Z=X.dot(self.W)+self.b
    return softmax(Z)
  
  def fit(self,X,Y,learning_rate=5*10e-7,reg=1.0,epochs=10000,show_fig=False):
    X,Y=shuffle(X,Y)
    Xtrain,Ytrain=X[:-1000],Y[:-1000]
    Xtest,Ytest=X[-1000:],Y[-1000:]
    
    N,D=X.shape
    K=len(set(Y))
    self.W=np.random.randn(D,K)
    self.b=np.zeros(K)
    T=y2indicator(Ytrain)
    Ttest=y2indicator(Ytest)
    
    costs=[]
    for i in range(epochs):
      Y_soft=self.forward(Xtrain)
      
      self.W-=learning_rate*(Xtrain.T.dot(Y_soft-T)+self.W)
      self.b-=learning_rate*((Y_soft-T).sum(axis=0)+reg*self.b)
      
      if i%1000==0:
        pYtest=self.forward(Xtest)

        c=cost(Ttest,pYtest)
        costs.append(c)
        e=errorRate(Ytest,np.argmax(pYtest,axis=1))
        print("i:",i,"cost",c,"error",e)
        
    
    if show_fig:
      plt.plot(costs)

  def predict(self,X):
    pY=self.forward(X)
    return np.argmax(pY,axis=1)
    
  def score(self,X,Y):
    pY=self.predict(X)
    return 1-errorRate(Y,pY)
    
def main():
  X,Y=getData()
  model = LogisticModel()
  model.fit(X,Y,show_fig=True)
  print(model.score(X,Y))
  
if __name__=='__main__':
  main()