import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(val):
    return 1/(1+np.exp(-val))

def calculate_cost(theta, X, y):
    return (-(y.transpose()@np.log(sigmoid(X@theta))) - ((1-y).transpose()@np.log(1-sigmoid(X@theta))))/X.shape[0];

def gradient_descent(theta, X, y, alpha):
    return theta - alpha*(X.transpose()@(sigmoid(X@theta) - y))/X.shape[0]
    
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

theta = np.array((0,0,0)).reshape(-1,1)
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