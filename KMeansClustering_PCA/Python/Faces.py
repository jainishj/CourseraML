import pandas as pd
import numpy as np
from numpy.linalg import svd

def feature_normalize(X):
    return (X-np.mean(X, axis = 0))/np.std(X, axis = 0)

def project_data(X, U, K):
    return X@U[:,0:K]

def recover_data(Z, U, K):
    return Z@U[:,0:K].T

faces_df = pd.read_csv('data/Faces.csv', header = None).values

X = feature_normalize(faces_df)


cov_matrix = X.T@X/len(X)

u,s,_ = svd(cov_matrix)
K = 100
Z = project_data(X, u, K)

X_Rec = recover_data(Z, u, K)