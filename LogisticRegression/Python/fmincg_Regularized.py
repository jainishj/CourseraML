import numpy as np
from scipy import optimize
import pandas as pd
from datetime import datetime
from sklearn import metrics

def sigmoid(val):
    return 1/(1+np.exp(-val))

def cost(theta, *args):
    X, y, lambda_ = args
    datasize = len(X)
    cost = (-(y.transpose()@np.log(sigmoid(X@theta))) - ((1-y).transpose()@np.log(1-sigmoid(X@theta))))/datasize + lambda_*(sum(theta[1:-1]**2))/(2*datasize);
    return cost[0]

def grad(theta, *args):
    theta = theta.reshape(-1,1)
    X, y, lambda_ = args
    
    datasize = len(X)
    gg = (X.transpose()@(sigmoid(X@theta) - y))/datasize
    gg[1:-1] += theta[1:-1]*lambda_/datasize
    return gg.flatten()

df = pd.read_csv('data/Microchip_Transformed.csv')

features = df[df.columns[0:-1]]
target = df[df.columns[-1]]

theta = np.array([0.0]*(len(features.columns))).reshape(-1,1)
  
startTime = datetime.now()
lambda_ = 1

#Call fmin cg function
res1= optimize.fmin_cg(cost, theta, fprime=grad, args=(features.values, target.values.reshape(-1,1), lambda_))

print('Time taken: {}'.format( datetime.now() - startTime))

#Predictions on parameters
predicted_target = np.round(sigmoid(features@res1))

print ('Theta after {} fmincg'.format(res1.tolist()))

#R2 score
print('R2_score:{}'.format(metrics.r2_score(target, predicted_target)))

cf_matrix = metrics.confusion_matrix(target, predicted_target)

print('F1 score:{}'.format(metrics.f1_score(target, predicted_target)))