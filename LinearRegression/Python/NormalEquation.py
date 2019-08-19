import pandas as pd
import numpy as np

dataset = pd.read_csv('data/MultiVarData.txt', header = None)
    
X = dataset.iloc[:,0:-1].values
Y = dataset.iloc[:,-1].values[:,np.newaxis]

feature_set = np.hstack((np.ones([X.shape[0],1]), X))
theta = np.linalg.inv(feature_set.transpose()@feature_set)@feature_set.transpose()@Y

X_house = np.array([1,1650,3]).reshape(1,3)
print('Cost of a new house:', X_house@theta)