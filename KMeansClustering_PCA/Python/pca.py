import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

def feature_normalize(X):
    return (X-np.mean(X, axis = 0))/(np.std(X, axis = 0))

def project_data(X, U, K):
    return X@U[:,0:K]

def recover_data(Z, U, K):
    return Z@U[:,0:K].T

random_df = pd.read_csv('data/pca1.csv', header = None).values

X = feature_normalize(random_df)

fig, ax = plt.subplots()
ax.plot(X[:,0], X[:,1], 'o')

cov_matrix = X.T@X/len(X)

u,s,_ = svd(cov_matrix)


Z = project_data(X, u, 1)

X_Rec = recover_data(Z, u, 1)
ax.plot(X_Rec[:,0], X_Rec[:,1], 'ro')
plt.show()