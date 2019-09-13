import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from datetime import datetime
from sklearn import linear_model
    
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
lm = linear_model.LinearRegression()
startTime = datetime.now()
#Fit inputs
lm.fit(population.values.reshape(-1,1), profits)
print('Time taken: {}'.format( datetime.now() - startTime))

print('Paramters from Linear Regressor Model:[{},{}]'.format(lm.intercept_, lm.coef_))

#Predict Outputs
predicted_profit = lm.predict(population.values.reshape(-1,1))

#Plot Regression line
foodtruck_df.plot(kind = 'scatter', x = 'Population', y = 'Profits')
plt.plot(population, predicted_profit, color = 'red')
plt.title('Profits vs Population')
plt.show()

#R2 score
print('R2_score:{}'.format(metrics.r2_score(profits, predicted_profit)))