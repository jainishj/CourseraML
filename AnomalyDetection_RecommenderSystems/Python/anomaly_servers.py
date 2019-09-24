import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import f1_score

def guassian_params(X):
    mean = np.mean(X, axis = 0)
    sigma = sum((X - mean)**2)/len(X)
    return mean, sigma

def guassian_val(X, mu, sigma):
    return scipy.stats.norm(mu[0], np.sqrt(sigma[0])).pdf(X[:,0]) * scipy.stats.norm(mu[1], np.sqrt(sigma[1])).pdf(X[:,1])

def bestEpsilon(X, y, mu, sigma):
    pval = guassian_val(X, mu, sigma)
    stepsize = (np.max(pval) - np.min(pval))/1000
    epsilons = np.arange(np.min(pval), np.max(pval), stepsize)
    best_eps = epsilons[0]
    best_f1score = 0
    for eps in epsilons:
        y_pred = (pval < eps).astype(int)
        score = f1_score(y, y_pred)
        if(score > best_f1score):
            best_f1score = score
            best_eps = eps
            
    return best_eps, eps

    
server_train = pd.read_csv('data/server_training.csv', header = None).values

fig, ax = plt.subplots()
ax.plot(server_train[:,0], server_train[:,1], 'bx')

server_val = pd.read_csv('data/server_val.csv', header = None).values

Xval = server_val[:,:-1]
yval = server_val[:,-1]
fig, ax = plt.subplots()
ax.plot(Xval[:,0], Xval[:,1], 'bx')


mu, sigma = guassian_params(server_train)
p_train = guassian_val(server_train, mu, sigma)
eps, f1score = bestEpsilon(Xval, yval, mu, sigma)

print('EPS Values:{} with f1 score:{}'.format(eps, f1score))

p_val = guassian_val(Xval, mu, sigma)
X_anomalous = Xval[p_val < eps]

ax.plot(X_anomalous[:,0], X_anomalous[:,1], 'rx')
plt.show()