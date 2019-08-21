import numpy as np
from scipy import optimize
import pandas as pd

def sigmoid(val):
    return 1/(1+np.exp(-val))

def cost(theta, *args):
    X, y = args
    cost = (-(y.transpose()@np.log(sigmoid(X@theta))) - ((1-y).transpose()@np.log(1-sigmoid(X@theta))))/X.shape[0];
    return cost[0]

def grad(theta, *args):
    theta = theta.reshape(-1,1)
    X, y = args
    gg = (X.transpose()@(sigmoid(X@theta) - y))/X.shape[0]
    return gg.flatten()

dataset = pd.read_csv('data/ex2data1.txt', header = None)

scores = dataset.iloc[:,0:-1].values
admitted = dataset.iloc[:,-1].values.reshape(-1,1)
training_size = scores.shape[0]
feature_set = np.hstack((np.ones([training_size,1]), scores))
theta = np.asarray((0,0,0)).reshape(-1,1) # Initial theta.
  
res1 = optimize.fmin_cg(cost, theta, fprime=grad, args=(feature_set, admitted))

print (res1)