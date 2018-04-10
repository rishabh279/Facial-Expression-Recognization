# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 19:01:12 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import softmax,cost,getData,errorRate,y2indicator

class AnnModel(object):
  def __init__(self,M):
    self.M=M
  
  def forward(self,X):
    Z=np.tanh(X.dot(self.W1)+self.b1)
    return softmax(Z.dot(self.W2)+self.b2),Z
    
  def fit(self,X,Y,learning_rate=5*10e-7,reg=1.0,epochs=10000,show_fig=False):
    X,Y=shuffle(X,Y)
    
    Xtrain,Ytrain=X[:-1000],Y[:-1000]
    Xtest,Ytest=X[-1000:],Y[-1000:]
    Ytrain_ind=y2indicator(Ytrain)
    Ytest_ind=y2indicator(Ytest)
    
    
    N,D=X.shape
    K=len(set(Y))
    self.W1=np.random.randn(D,self.M)
    self.b1=np.random.randn(self.M)
    self.W2=np.random.randn(self.M,K)
    self.b2=np.random.randn(K)
    
    costs=[]
    for i in range(epochs):
      pY,Z=self.forward(Xtrain)
      self.W2-=learning_rate*(Z.T.dot(pY-Ytrain_ind)+reg*self.W2)
      self.b2 -= learning_rate*((pY-Ytrain_ind).sum(axis=0) + reg*self.b2)
      
      
      dZ = (pY-Ytrain_ind).dot(self.W2.T) * (1 - Z*Z) # tanh
      self.W1 -= learning_rate*(Xtrain.T.dot(dZ) + reg*self.W1)
      self.b1 -= learning_rate*(dZ.sum(axis=0) + reg*self.b1)
      
      if(i%1000==0):
        pYtest,_=self.forward(Xtest)
        c=cost(Ytest_ind,pYtest)
        costs.append(c)
        e=errorRate(Ytest,np.argmax(pYtest,axis=1))
        print("i:",i,"cost",c,"error",e)      
      if show_fig==True:
        plt.plot(costs)
        
  def predict(self,X):
    pY,_=self.forward(X)
    return np.argmax(pY,axis=1)  
    
  def score(self,X,Y):
    prediction=self.predict(X)
    return 1-errorRate(Y,prediction)
  
def main():
  X,Y=getData()
  model = AnnModel(200)
  model.fit(X,Y,reg=0,show_fig=True)
  print(model.score(X,Y))

if __name__=='__main__':
  main()
