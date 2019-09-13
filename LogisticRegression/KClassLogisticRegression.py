import numpy as np
from scipy import optimize
import pandas as pd

def sigmoid(val):
    return 1/(1+np.exp(-val))

def cost(theta, *args):
    X, y = args
    cost = (-(y.transpose()@np.log(sigmoid(X@theta))) - ((1-y).transpose()@np.log(1-sigmoid(X@theta))))/X.shape[0];
    return cost[0]

def grad(theta, *args):
    theta = theta.reshape(-1,1)
    X, y = args
    gg = (X.transpose()@(sigmoid(X@theta) - y))/X.shape[0]
    return gg.flatten()

pixel_data = pd.read_csv('data/pixel_data.csv', header = None)
y_data = pd.read_csv('data/pixel_data_results.csv', header = None)

input_ = pixel_data.values
output_ = y_data.values.reshape(-1,1)
training_size = input_.shape[0]
feature_set = np.hstack((np.ones([training_size,1]), input_))

results = np.zeros((10,401), dtype =float)

for c in range(1,11):
    out = (output_ == c).astype(int)
    theta = np.zeros((feature_set.shape[1],1)) # Initial theta.
    res = (optimize.fmin_cg(cost, theta, fprime=grad, args=(feature_set, out)))
    results[c-1] = res.flatten()

predictions = np.argmax(feature_set@results.transpose(), axis = 1);

quality = (output_ == (predictions.reshape(-1,1)+1))
print (np.count_nonzero(quality == True)/5000*100)
