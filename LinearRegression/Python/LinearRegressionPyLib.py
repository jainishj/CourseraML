import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/MultiVarData.txt', header = None)
    
X =  dataset.iloc[:,0].values.reshape(-1,1)
Y = dataset.iloc[:,1].values.reshape(-1,1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)
coef = regressor.coef_
intercept = regressor.intercept_
# Visualising the Training set results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Population vs Profit')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()