import cv2
import numpy as np
from sklearn.metrics.pairwise import paired_distances
def findClosestCentroids(X, centroids):
    idx = np.zeros((len(X),))
    for i in range(len(X)):
        x = np.repeat(X[i].reshape(-1,1), len(centroids), axis = 1).T
        eucl_dist = paired_distances(x, centroids)
        idx[i] = np.argmin(eucl_dist)
    return idx

def computeMeans(X, idx, K):
    mean_centroids = np.zeros((K, X.shape[1]))
    for i in range(K):
        temp = X[idx == i]
        if(temp.size == 0):
            temp = np.array([0,0,0])
        mean_centroids[i,:] = np.mean(temp, axis =0)
    return mean_centroids
    
img = cv2.imread('data/bird_small.png')
img_flat = img/255.0

img_flat = np.reshape(img_flat, (img.shape[0]*img.shape[1],3))

K=16
#initialize random centroids
centroids = np.random.rand(K, img_flat.shape[1])
prev_centroids = np.zeros(centroids.shape)
for i in range(25):
    prev_centroids = centroids.copy()
    idx = findClosestCentroids(img_flat, centroids)
    centroids = computeMeans(img_flat, idx, K)
    print(centroids[0:4,:])
    
idx = idx.astype(np.ushort)
l = centroids[idx,:]*255
l = np.reshape(l, (img.shape[0], img.shape[1], 3))

cv2.imwrite('output.png',l)