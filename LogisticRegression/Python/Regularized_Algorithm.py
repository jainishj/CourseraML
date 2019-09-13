import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import metrics

def sigmoid(x):
    return 1/(1+np.exp(-x))

#Cost function
def costFunc(X, y, theta, reg = 0.0):
    datasize = len(X)
    
    #Append bias vector of 1s
    X = np.insert(X.values, 0, np.ones(datasize,), axis = 1)
    cost = 0
    for i in range(datasize):
        predicted_cost = 0
        for j in range(len(theta)):
            predicted_cost += X[i,j]*theta[j] 
        cost += -(y[i]*np.log(sigmoid(predicted_cost)) + (1-y[i])*np.log(1-sigmoid(predicted_cost)))
    
    reg_term = 0
    for i in range(1,len(theta)):
        reg_term += theta[i]**2
    return cost/datasize + reg*reg_term/(2*datasize)

#Gradient Descent
def gradientDescent(X, y, theta, alpha, reg = 0.0):
    datasize = len(X)
    
    #Append bias vector of 1s
    X = np.insert(X.values, 0, np.ones(datasize,), axis = 1)
    derivative = [0]*len(theta)
    
    #Calculate Derivative Value
    #for i in range(len(theta)):
    for j in range(datasize):
        predicted_cost = 0
        for k in range(len(theta)):
            predicted_cost += X[j,k]*theta[k]
        
        derivative[0] += (sigmoid(predicted_cost) - y[j])*X[j,k]
        for k in range(1, len(theta)):
            derivative[k] += (sigmoid(predicted_cost) - y[j])*X[j,k] + reg*theta[k]/datasize
            
    for i in range(len(theta)):
        theta[i] = theta[i] - derivative[i]*alpha/datasize

    return theta

df = pd.read_csv('data/Microchip_Transformed.txt')

features = df[df.columns[0:-1]]
target = df[df.columns[-1]]

#Cost at theta (0,0,0)
print('Cost at theta {}: {}'.format([0,0,0], costFunc(features, target, [0,0,0])))


startTime = datetime.now()

theta = np.array([0.0]*(len(features.columns)+1))
##theta = np.array([-3.63029144, 1.16636235])
cost = []
cost.append(costFunc(features, target, theta))
prev_cost = 0
lambda_ = 1
#Gradient Descent over multiple steps
#while(abs(cost[-1]-prev_cost) > 0.00000):
for i in range(1000):
    prev_cost = cost[-1]
    theta = gradientDescent(features, target, theta, alpha = 0.1, reg = lambda_)
    cost.append(costFunc(features, target, theta, reg = lambda_))

print('Time taken: {}'.format( datetime.now() - startTime))

plt.plot(cost)
plt.show()

print ('Theta after {} iterations with alpha {}: {}'.format(len(cost), 0.01, theta.tolist()))

predicted_target = []

for i in range(len(features)):
    target_ = theta[0]
    for j in range(len(theta)-1):
        target_ += features.iloc[i][j]*theta[j+1]
    predicted_target.append(round(sigmoid(target_)))
    
#R2 score
print('R2_score:{}'.format(metrics.r2_score(target, predicted_target)))

cf_matrix = metrics.confusion_matrix(target, predicted_target)

print('F1 score:{}'.format(metrics.f1_score(target, predicted_target)))