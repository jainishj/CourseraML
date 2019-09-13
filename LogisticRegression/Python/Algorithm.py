import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import metrics

def sigmoid(x):
    return 1/(1+np.exp(-x))

#Cost function
def costFunc(X, y, theta):
    datasize = len(X)
    
    #Append bias vector of 1s
    X = np.insert(X.values, 0, np.ones(datasize,), axis = 1)
    cost = 0
    for i in range(datasize):
        predicted_cost = 0
        for j in range(len(theta)):
            predicted_cost += X[i,j]*theta[j] 
        cost += -(y[i]*np.log(sigmoid(predicted_cost)) + (1-y[i])*np.log(1-sigmoid(predicted_cost)))
    return cost/datasize

#Gradient Descent
def gradientDescent(X, y, theta, alpha):
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
        for k in range(len(theta)):
            derivative[k] += (sigmoid(predicted_cost) - y[j])*X[j,k]
            
    for i in range(len(theta)):
        theta[i] = theta[i] - derivative[i]*alpha/datasize

    return theta

df = pd.read_csv('data/StudentGrades.csv', names = ['Exam1', 'Exam2', 'Admitted'])

df_pos = df[df['Admitted'] == 1]
df_neg = df[df['Admitted'] == 0]

ax = df_pos.plot(kind = 'scatter', x = 'Exam1', y = 'Exam2', c = 'green')
df_neg.plot(kind = 'scatter', x = 'Exam1', y = 'Exam2', c = 'red', ax = ax)
plt.title('Grades vs Admission Status')
plt.show()

features = df[df.columns[0:-1]]
target = df[df.columns[-1]]

#Feature Normalization
features_norm = (features - features.mean())/features.std()

#Cost at theta (0,0,0)
print('Cost at theta {}: {}'.format([0,0,0], costFunc(features_norm, target, [0,0,0])))


startTime = datetime.now()

theta = np.array([0.0]*(len(features.columns)+1))
##theta = np.array([-3.63029144, 1.16636235])
cost = []
cost.append(costFunc(features_norm, target, theta))
prev_cost = 0

#Gradient Descent over multiple steps
while(abs(cost[-1]-prev_cost) > 0.000001):
#for i in range(400):
    prev_cost = cost[-1]
    theta = gradientDescent(features_norm, target, theta, alpha = 0.1)
    cost.append(costFunc(features_norm, target, theta))

print('Time taken: {}'.format( datetime.now() - startTime))

plt.plot(cost)
plt.show()

print ('Theta after {} iterations with alpha {}: {}'.format(len(cost), 0.01, theta.tolist()))

predicted_target = []

for i in range(len(features)):
    target_ = theta[0]
    for j in range(len(theta)-1):
        target_ += features_norm.iloc[i][j]*theta[j+1]
    predicted_target.append(round(sigmoid(target_)))
    
#R2 score
print('R2_score:{}'.format(metrics.r2_score(target, predicted_target)))

cf_matrix = metrics.confusion_matrix(target, predicted_target)

print('F1 score:{}'.format(metrics.f1_score(target, predicted_target)))