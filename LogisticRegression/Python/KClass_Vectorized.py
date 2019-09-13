import pandas as pd
from scipy import optimize
import numpy as np
from sklearn import metrics
from datetime import datetime

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

def trainLogisticRegressionModel(X, y, theta):
    startTime = datetime.now()
    lambda_ = 0

    #Call fmin cg function
    res1= optimize.fmin_cg(cost, theta, fprime=grad, args=(X.values, y.values.reshape(-1,1), lambda_))
    
    print('Time taken: {}'.format( datetime.now() - startTime))
    
    print ('Theta after {} fmincg'.format(res1.tolist()))
    return res1

#Read Data
df = pd.read_csv('data/Numbers_Pixel.csv')

thetas = []
#Extract features and target from dataframe
features = df[df.columns[0:-1]]
for K in range(1,11):
    target = df.iloc[:,-1] == K
    target = target.astype(int)
    theta = trainLogisticRegressionModel(features, target, np.array([0.0]*(len(features.columns))))
    thetas.append([K, theta])

predicted_probability = []
for i in range(len(features)):
    for theta in thetas:
        predicted_probability.append([i, [theta[0], sigmoid(features[i]@theta[1].T)]])

#R2 score
#print('R2_score:{}'.format(metrics.r2_score(target, predicted_target)))

#cf_matrix = metrics.confusion_matrix(target, predicted_target)

#print('F1 score:{}'.format(metrics.f1_score(target, predicted_target)))