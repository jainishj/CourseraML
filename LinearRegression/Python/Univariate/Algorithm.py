import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from datetime import datetime

#Cost function
def costFunc(X, y, theta):
    datasize = len(X)
    
    #Append bias vector of 1s
    X = np.stack((np.ones(datasize,), X.values), axis = -1)
    cost = 0
    for i in range(datasize):
        cost += (X[i,0]*theta[0] + X[i,1]*theta[1] - y[i])**2
    return cost/(2*datasize)

#Gradient Descent
def gradientDescent(X, y, theta, alpha):
    datasize = len(X)
    
    #Append bias vector of 1s
    X = np.stack((np.ones(datasize,), X.values), axis = -1)
    derivative = [0,0]
    
    #Calculate Derivative Value
    for i in range(datasize):
        derivative[0] += (X[i,0]*theta[0] + X[i,1]*theta[1] - y[i])*X[i,0]
        derivative[1] += (X[i,0]*theta[0] + X[i,1]*theta[1] - y[i])*X[i,1]
    for i in range(len(theta)):
        theta[i] = theta[i] - derivative[i]*alpha/datasize
    return theta

#Read data
foodtruck_df = pd.read_csv('data/FoodTruckProfits.csv', names = ['Population', 'Profits'])

#Extract Population Data
population = foodtruck_df['Population']

#Extract Profit Data
profits = foodtruck_df['Profits']

#Plot Profit vs Population
foodtruck_df.plot(kind = 'scatter', x = 'Population', y = 'Profits')
plt.title('Profits vs Population')
plt.show()

#Cost at theta (0,0) & (-1,2)
print('Cost at theta: {}.{}: {}'.format(0, 0, costFunc(population, profits, [0,0])))
print('Cost at theta: {}.{}: {}'.format(-1, -2, costFunc(population, profits, [-1,2])))

startTime = datetime.now()

theta = np.array([0.0,0.0])
#theta = np.array([-3.63029144, 1.16636235])
cost = []
cost.append(costFunc(population, profits, theta))
prev_cost = 0

#Gradient Descent over multiple steps
while(abs(cost[-1]!=prev_cost) > 0.00001):
#for i in range(1500):
    prev_cost = cost[-1]
    theta = gradientDescent(population, profits, theta, alpha = 0.01)
    cost.append(costFunc(population, profits, theta))

print('Time taken: {}'.format( datetime.now() - startTime))

plt.plot(cost)
plt.show()

print ('Theta after {} iterations with alpha {}: {}'.format(len(cost), 0.01, theta.tolist()))

#Plot regression line 
foodtruck_df.plot(kind = 'scatter', x = 'Population', y = 'Profits')
plt.title('Profits vs Population')
plt.plot(population, theta[0]*1 + theta[1]*population)
plt.show()

predicted_profit = []

for i in range(len(population)):
    predicted_profit.append(theta[0] + population[i]*theta[1])
    
#R2 score
print('R2_score:{}'.format(metrics.r2_score(profits, predicted_profit)))
