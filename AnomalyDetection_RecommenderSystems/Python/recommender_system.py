import numpy as np
import pandas as pd

def costFunc(u_w, m_w, n_m, n_u, R, Y, reg = 0):
    cost = np.sum(((m_w@u_w.T) * R - Y)**2)/2 + reg * np.sum(m_w**2)/2 + reg * np.sum(u_w**2)/2
    return cost

def gradientDescent(u_w, m_w, n_m, n_u, R, Y, reg = 0):
    movie_grad = (m_w@u_w.T - Y)*R@u_w + reg * m_w
    user_grad = ((m_w@u_w.T - Y)*R).T@m_w + reg * u_w
    
R = pd.read_csv('data/R.csv', header = None).values
Y = pd.read_csv('data/Y.csv', header = None).values

print('Average Rating for Movie 1(Toy Story):', np.mean(Y[0,R[0,:] == 1]))

user_weights = pd.read_csv('data/user_weights.csv', header = None).values
movie_weights = pd.read_csv('data/movie_weights.csv', header = None).values

num_movies = Y.shape[0]
num_users = Y.shape[1]

num_users = 4; num_movies = 5; num_features = 3;
movie_weights = movie_weights[0:num_movies, 0:num_features];
user_weights = user_weights[0:num_users, 0:num_features];
Y = Y[0:num_movies, 0:num_users];
R = R[0:num_movies, 0:num_users];

print('Cost at loaded parameters:', costFunc(user_weights, movie_weights, num_movies, num_users, R, Y))
print('Cost at loaded parameters with reg 1.5:', costFunc(user_weights, movie_weights, num_movies, num_users, R, Y, 1.5))