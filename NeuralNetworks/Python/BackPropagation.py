import pandas as pd
import numpy as np

def sigmoid(val):
    return 1/(1+np.exp(-val))

def randInit(rows, columns, eps_init = 0.12):
    return np.random.rand(rows, columns)*2*eps_init - eps_init

def cost(X, y, num_labels, theta1, theta2):
    m = X.shape[0]
    for i in range(m):
        a1 = X
    return
    
def activationUnits(weights, input_):
    input_ = np.hstack((np.ones([input_.shape[0],1]), input_))
    return sigmoid(input_@weights.transpose())

pixel_data = pd.read_csv('data/pixel_data.csv', header = None).values
results = pd.read_csv('data/pixel_data_results.csv', header = None).values
num_labels = 10
m = pixel_data.shape[0]
y = np.zeros((m, num_labels))

for i in range(m):
    y[i,results[i]-1] = 1
    
theta1 = randInit(401, 25)
theta2 = randInit(26, 10)


    