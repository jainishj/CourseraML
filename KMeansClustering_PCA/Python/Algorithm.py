import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import paired_distances

random_df = pd.read_csv('data/random.csv')
X = random_df.values

fig, ax = plt.subplots()
ax.plot(X[:,0], X[:,1], 'o')

def findClosestCentroids(X, centroids):
    idx = np.zeros((len(X),))
    for i in range(len(X)):
        x = np.repeat(X[i].reshape(-1,1), len(centroids), axis = 1).T
        eucl_dist = paired_distances(x, centroids)
        idx[i] = np.argmin(eucl_dist)
    return idx

def computeMeans(X, idx):
    return np.array([np.mean(X[idx == 0], axis = 0),
                     np.mean(X[idx == 1], axis = 0),
                     np.mean(X[idx == 2], axis = 0)])

#initialize random centroids
centroids = np.random.rand(3,2)*np.array([np.max(X[:,0]),np.max(X[:,1])])
prev_centroids = np.zeros(centroids.shape)
while((prev_centroids == centroids).all() == False):
    prev_centroids = centroids.copy()
    idx = findClosestCentroids(X, centroids)
    centroids = computeMeans(X, idx)
    fig, ax1 = plt.subplots()
    
    ax1.plot(X[idx == 0][:,0], X[idx == 0][:,1],'o')
    ax1.plot(X[idx == 1][:,0], X[idx == 1][:,1], 'go')
    ax1.plot(X[idx == 2][:,0], X[idx == 2][:,1], 'ro')
    
    ax1.plot(centroids[0,0], centroids[0,1], 'b', marker = 'o', markersize=8)
    ax1.plot(centroids[1,0], centroids[1,1], 'springgreen', marker = 'o', markersize=8)
    ax1.plot(centroids[2,0], centroids[2,1], 'brown', marker = 'o', markersize=8)
    
    plt.show()