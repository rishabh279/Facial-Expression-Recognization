# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 12:43:58 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 
from util import getBinaryData,relu,sigmoidCost,sigmoid,errorRate

class ANN(object):
  def __init__(self,M):
    self.M=M
    
  def forward(self,X):
    Z=np.tanh(X.dot(self.W1)+self.b1)
    return sigmoid(Z.dot(self.W2)+self.b2),Z
   
    
  def fit(self,X,Y,learning_rate=5*10e-7,reg=1.0,epochs=10000,show_fig=False):
    X,Y=shuffle(X,Y)
    Xvalid,Yvalid=X[-1000:],Y[-1000:]
    X,Y=X[:-1000],Y[:-1000]
      
    N,D=X.shape
    self.W1=np.random.randn(D,self.M)/np.sqrt(D)
    self.b1=np.zeros(self.M)
    self.W2=np.random.randn(self.M)/np.sqrt(self.M)
    self.b2=0
    
    costs=[]
    for i in range(epochs):
      pY,Z=self.forward(X)  
      
      self.W2-=learning_rate*(Z.T.dot(pY-Y)+reg*self.W2)
      self.b2-=learning_rate*((pY-Y).sum()+reg*self.b2)
      
      dZ=np.outer(pY-Y,self.W2)*(1-Z*Z)
      self.W1-=learning_rate*(X.T.dot(dZ)+reg*self.b1)
      self.b1-=learning_rate*(np.sum(dZ,axis=0)+reg*self.b1)
      
      if i%100==0:
        pYvalid,_=self.forward(Xvalid)
        c=sigmoidCost(Yvalid,pYvalid)
        costs.append(c)
        e=errorRate(Yvalid,np.round(pYvalid))
        print("i:",i,"cost:",c,"error:",e)
        
      if show_fig:
        plt.plot(costs)
        
        
    def predict(self,X):
      pY=self.forward(X)
      return np.round(pY)
      
    def score(self,X,Y):
      prediction=self.predict
      return 1-errorRate(Y,prediction)

def main():
  X,Y=getBinaryData()
   
  X0=X[Y==0,:]
  X1=X[Y==1,:]
  X1=np.repeat(X1,9,axis=0)
  X=np.vstack([X0,X1])
  Y=np.array([0]*len(X0)+[1]*len(X1))
  model = ANN(100)
  model.fit(X,Y,show_fig=True)

if __name__=='__main__':
  main()
  
        