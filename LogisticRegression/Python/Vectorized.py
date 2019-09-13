import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from datetime import datetime

def sigmoid(x):
    return 1/(1+np.exp(-x))

#Cost function
def costFunc(X, y, theta):
    datasize = len(X)
    
    #Append bias vector of 1s
    X = np.insert(X.values, 0, np.ones(datasize,), axis = 1)
    return sum(-(y*np.log(sigmoid(X@theta.T)) + (1-y)*np.log(1- sigmoid(X@theta.T))))/datasize

#Gradient Descent
def gradientDescent(X, y, theta, alpha):
    datasize = len(X)
    
    #Append bias vector of 1s
    X = np.insert(X.values, 0, np.ones(datasize,), axis = 1)
    derivative = (X.T@(sigmoid(X@theta.T) - y))
    theta = theta - derivative*alpha/datasize
    return theta
    
#Read Data
df = pd.read_csv('data/StudentGrades.csv', names = ['Exam1', 'Exam2', 'Admitted'])

#Split Positive & Negative Data
df_pos = df[df['Admitted'] == 1]
df_neg = df[df['Admitted'] == 0]

#Plot
ax = df_pos.plot(kind = 'scatter', x = 'Exam1', y = 'Exam2', c = 'green')
df_neg.plot(kind = 'scatter', x = 'Exam1', y = 'Exam2', c = 'red', ax = ax)
plt.title('Grades vs Admission Status')
plt.show()

#Extract features and target from dataframe
features = df[df.columns[0:-1]]
target = df[df.columns[-1]]

#Feature Normalization
features_norm = (features - features.mean())/features.std()
  
#Cost at theta (0,0,0)
print('Cost at theta {}: {}'.format([0,0,0], costFunc(features_norm, target, np.array([0,0,0]))))

startTime = datetime.now()
theta = np.array([0.0]*(len(features.columns)+1))
cost = []
cost.append(costFunc(features_norm, target, theta))
prev_cost = 0

#Gradient Descent over multiple steps
while(abs(cost[-1]-prev_cost) > 0.000001):
#for i in range(1000):
    prev_cost = cost[-1]
    theta = gradientDescent(features_norm, target, theta, alpha = 0.01)
    cost.append(costFunc(features_norm, target, theta))

print('Time taken: {}'.format( datetime.now() - startTime))

plt.plot(cost)
plt.show()

#Predictions on parameters
predicted_target = np.round(sigmoid(np.insert(features_norm.values, 0, np.ones(len(features_norm),), axis = 1)@theta.T))

print ('Theta after {} iterations with alpha {}: {}'.format(len(cost), 0.01, theta.tolist()))

#R2 score
print('R2_score:{}'.format(metrics.r2_score(target, predicted_target)))

cf_matrix = metrics.confusion_matrix(target, predicted_target)

print('F1 score:{}'.format(metrics.f1_score(target, predicted_target)))