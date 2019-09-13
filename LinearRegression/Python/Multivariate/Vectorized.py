import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from datetime import datetime

#Cost function
def costFunc(X, y, theta):
    datasize = len(X)
    
    #Append bias vector of 1s
    X = np.insert(X.values, 0, np.ones(datasize,), axis = 1)
    return sum((X@theta.T - y)**2)/(2*datasize)

#Gradient Descent
def gradientDescent(X, y, theta, alpha):
    datasize = len(X)
    
    #Append bias vector of 1s
    X = np.insert(X.values, 0, np.ones(datasize,), axis = 1)
    derivative = X.T@(X@theta.T - y)
    theta = theta - derivative*alpha/datasize
    return theta
    
#Read data
df = pd.read_csv('data/HousingPrice.csv', names = ['Size', 'Bedrooms', 'Price'])

#Extract features
features = df[df.columns[0:-1]]

#Extract Target
target = df[df.columns[-1]]
target_name = target.name

for i in range(len(features.columns)):
    col_name = features.columns[i]
    df.plot(kind = 'scatter', x = col_name, y = target_name)
    plt.title('{} vs {}'.format(col_name, target_name))
    plt.show()

#Feature Normalization
features_norm = (features - features.mean())/features.std()
  
#Cost at theta (0,0,0)
print('Cost at theta {}: {}'.format([0,0,0], costFunc(features, target, np.array([0,0,0]))))

startTime = datetime.now()
theta = np.array([0.0]*(len(features.columns)+1))
cost = []
cost.append(costFunc(features_norm, target, theta))
prev_cost = 0

#Gradient Descent over multiple steps
#while(abs(cost[-1]!=prev_cost) > 0.00001):
for i in range(1000):
    prev_cost = cost[-1]
    theta = gradientDescent(features_norm, target, theta, alpha = 0.01)
    cost.append(costFunc(features_norm, target, theta))

print('Time taken: {}'.format( datetime.now() - startTime))

plt.plot(cost)
plt.show()

print ('Theta after {} iterations with alpha {}: {}'.format(len(cost), 0.01, theta.tolist()))

predicted_target = np.insert(features_norm.values, 0, np.ones(len(features_norm),), axis = 1)@theta.T
    
#R2 score
print('R2_score:{}'.format(metrics.r2_score(target, predicted_target)))

