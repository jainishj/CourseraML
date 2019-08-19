import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_cost(theta, X, Y):
    training_size = X.shape[0]
    feature_set = np.hstack((np.ones([training_size,1]), X))
    cost_vec = feature_set@theta - Y
    return np.sum(cost_vec**2/(2*training_size))
    
def gradient_descent(theta, X, Y, alpha):
    training_size = X.shape[0]
    feature_set = np.hstack((np.ones([training_size,1]),  X))
    derivates = feature_set.transpose()@(feature_set@theta - Y)
    val= theta - alpha * derivates/training_size
    return val

dataset = pd.read_csv('data/UniVarData.txt', header = None)
    
X = dataset.iloc[:,0:-1].values
Y = dataset.iloc[:,-1].values[:,np.newaxis]

if X.shape[1] > 1:
    X = X.astype(np.float)
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.mean(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))
    Y = (Y - np.mean(Y))/(np.max(Y) - np.min(Y))

plt.figure(0)
plt.plot(X, Y, 'ro')
plt.xlabel('Population')
plt.ylabel('profit')
plt.title('Population vs Profit')

theta = np.zeros([X.shape[1]+1,1])

cost = calculate_cost(theta, X, Y)
prev_cost = -1
cost_over_iterations = []
while prev_cost != cost:
    theta = gradient_descent(theta, X, Y, 0.01)
    prev_cost = cost
    cost = calculate_cost(theta, X, Y)
    cost_over_iterations.append(cost)
    
plt.figure(1)
plt.plot(cost_over_iterations)

    
predictions = np.hstack((np.ones([X.shape[0],1]),  X))@theta
plt.figure(0)
plt.plot(X, predictions)

print ('Iterations Required:',len(cost_over_iterations) )