import numpy as np
from scipy import optimize
import pandas as pd

def sigmoid(val):
    return 1/(1+np.exp(-val))

def cost(theta, *args):
    X, y, lambda_ = args
    train_size = X.shape[0]
    cost = (-(y.transpose()@np.log(sigmoid(X@theta))) - ((1-y).transpose()@np.log(1-sigmoid(X@theta))))/train_size + lambda_/(2*train_size)*sum(theta[2:-1]**2);
    return cost[0]

def grad(theta, *args):
    theta = theta.reshape(-1,1)
    X, y, lambda_ = args
    train_size = X.shape[0]
    gg = (X.transpose()@(sigmoid(X@theta) - y))/train_size + lambda_/train_size*theta
    gg[0] = sum(sigmoid(X@theta)-y)/train_size
    return gg.flatten()

dataset = pd.read_csv('data/ex2data3.txt', header = 1)
dataset2 = pd.read_csv('data/ex2data2.txt', header = None)
input_ = dataset.to_numpy()

training_size = input_.shape[0]
feature_set = np.hstack((np.ones([training_size,1]), input_))
output_ = dataset2.iloc[2:,-1].values.reshape(-1,1)

theta = np.zeros(29).reshape(-1,1) # Initial theta.
lambda_=1
res1 = optimize.fmin_cg(cost, theta, fprime=grad, args=(feature_set, output_, lambda_))

y_pred = feature_set@res1.reshape(-1,1) == output_