import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('data/Training.csv', header = None)
val_df = pd.read_csv('data/Validation.csv', header = None)
test_df = pd.read_csv('data/Test.csv', header = None)

X_train = train_df.iloc[:,:-1]
y_train = train_df.iloc[:,-1]

X_val = val_df.iloc[:,:-1]
y_val = val_df.iloc[:,-1]

X_test = test_df.iloc[:,:-1]
y_test = test_df.iloc[:,-1]

f, ax = plt.subplots(1)
ax.scatter( X_train, y_train)

polynomial_model = PolynomialFeatures(degree = 8)
X_train_poly = polynomial_model.fit_transform(X_train)
X_val_poly = polynomial_model.fit_transform(X_val)

linear_model = LinearRegression(normalize = True)
linear_model.fit(X_train_poly, y_train)
 
print('mean_squared_error on training data:', mean_squared_error(y_train, linear_model.predict(X_train_poly)))

print('mean_squared_error on validation data:', mean_squared_error(y_val, linear_model.predict(X_val_poly)))

ridge = Ridge(alpha = 0.001, normalize = True)
ridge.fit(X_train_poly, y_train)

print('mean_squared_error on training data:', mean_squared_error(y_train, ridge.predict(X_train_poly)))

print('mean_squared_error on validation data:', mean_squared_error(y_val, ridge.predict(X_val_poly)))
def func():
    yield np.arange(0,22), np.arange(23,33)
#Using Grid search to find best value for alpha
combined_X = np.vstack([X_train_poly, X_val_poly])
combined_y = pd.concat([y_train, y_val], axis = 0)
alphas = np.linspace(0.001,10,20)
ridge = Ridge()
grid = GridSearchCV(estimator=ridge, param_grid=dict(alpha=alphas), cv = func())
grid.fit(combined_X, combined_y)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha)