import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(val):
    return 1/(1+np.exp(-val))

def calculate_cost(theta, X, y):
    total_cost = 0
    training_size = X.shape[0]
    for i in range(training_size):
        h_X = sigmoid(theta[0]*X[i][0] + theta[1]*X[i][1] + theta[2]*X[i][2])
        total_cost += (-y[i]*np.log(h_X)-(1-y[i])*np.log(1-h_X))
    return total_cost/training_size

def gradient_descent(theta, X, y, alpha):
    derivative = [0,0,0]
    training_size = X.shape[0]
    for i in range(training_size):
        h_X = sigmoid(theta[0]*X[i][0] + theta[1]*X[i][1] + theta[2]*X[i][2])
        for j in range(len(theta)):
            derivative[j] += (h_X-y[i])*X[i][j]
    return ((theta[0] - alpha*derivative[0]/training_size), 
            (theta[1] - alpha*derivative[1]/training_size),
            (theta[2] - alpha*derivative[2]/training_size));
    
    
dataset = pd.read_csv('data/ex2data1.txt', header = None)

scores = dataset.iloc[:,0:-1].values
admitted = dataset.iloc[:,-1].values.reshape(-1,1)
training_size = scores.shape[0]
feature_set = np.hstack((np.ones([training_size,1]), scores))

plt.figure(0)
plt.scatter(scores[:,0], scores[:,1], c = admitted[:,0])
plt.xlabel('Score 1')
plt.ylabel('Score 2')
plt.title('Admission vs Scores')
plt.show()

theta = (0,0,0)
previous_cost = -1;
cost = calculate_cost(theta, feature_set, admitted)
cost_over_iteration = []
alpha = .001
while previous_cost != cost:
    theta = gradient_descent(theta, feature_set, admitted, alpha)
    previous_cost = cost;
    cost = calculate_cost(theta, feature_set, admitted)
    cost_over_iteration.append(cost)
    print(len(cost_over_iteration), cost)
    
plt.figure(1)
plt.plot(cost_over_iteration)

predictions = feature_set@theta >= 0.5