import pandas as pd
import numpy as np

def sigmoid(val):
    return 1/(1+np.exp(-val))

def activationUnits(weights, input_):
    input_ = np.hstack((np.ones([input_.shape[0],1]), input_))
    return sigmoid(input_@weights.transpose())

pixel_data = pd.read_csv('data/pixel_data.csv', header = None).values
results = pd.read_csv('data/pixel_data_results.csv', header = None).values
theta1 = pd.read_csv('data/Theta1.csv', header = None).values
theta2 = pd.read_csv('data/Theta2.csv', header = None).values

a1 = activationUnits(theta1, pixel_data)
y = activationUnits(theta2, a1)

predictions = np.argmax(y, axis = 1);

quality = (results == (predictions.reshape(-1,1))+1)
print (np.count_nonzero(quality == True)/5000*100)