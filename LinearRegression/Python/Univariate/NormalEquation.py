import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from datetime import datetime
import numpy as np
    
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

#Create Linear Regressor

startTime = datetime.now()

#Normal Equation
X = np.stack((np.ones(len(population),), population.values), axis = -1)
theta = np.linalg.pinv(X.T@X)@X.T@profits
print('Time taken: {}'.format( datetime.now() - startTime))

print ('Theta calculated from normal equation: {}'.format(theta.tolist()))


#Plot Regression line
foodtruck_df.plot(kind = 'scatter', x = 'Population', y = 'Profits')
plt.plot(population, theta[0]*1 + theta[1]*population, color = 'red')
plt.title('Profits vs Population')
plt.show()


predicted_profit = np.stack((np.ones(len(population),), population.values), axis = -1)@theta.T

#R2 score
print('R2_score:{}'.format(metrics.r2_score(profits, predicted_profit)))