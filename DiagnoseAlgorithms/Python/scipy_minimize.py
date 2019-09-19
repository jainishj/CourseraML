import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def cost(theta, *args):
    X, y, lambda_ = args
    datasize = len(X)
    #Add Bias unit
    X = np.insert(X.values,0,np.ones(datasize), axis = 1)
    return np.sum((X@theta-y)**2)/(2*datasize) + lambda_*np.sum(theta[1:]**2)/(2*datasize)

def gradientDescent(theta, *args):
    X, y, lambda_ = args
    datasize = len(X)
    #Add Bias unit
    X = np.insert(X.values,0,np.ones(datasize), axis = 1)
    theta_reg = theta.copy()
    theta_reg[0] = 0
    return X.T@(X@theta-y)/datasize + lambda_*theta_reg/datasize

def polynomialFeatures(X, degree):
    res = np.zeros((len(X), degree))
    for i in range(degree):
        res[:,i] = (X**(i+1)).iloc[:,0] 
    return pd.DataFrame(res)

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

print('Cost with theta[1,1] with lambda 1:', cost(np.array([1,1]), X_train, y_train, 1))
print('Gradient with theta[1,1] with lambda 1:', gradientDescent(np.array([1,1]), X_train, y_train, 1))

res = optimize.minimize(cost, np.array([0,0]), args = (X_train, y_train, 0), method = 'CG', jac = gradientDescent, options = {'disp' :False})
ax.plot(X_train, res.x[0] + res.x[1]*X_train)
plt.show()

errors = {'train':[], 'val':[]}
for i in range(len(X_train)):
    theta = np.array([0,0])
    res1 = optimize.minimize(cost, theta, args = (X_train[:i+1], y_train[:i+1], 0), method = 'CG', jac = gradientDescent, options = {'disp' :False})
    errors['train'].append(res1.fun)
    errors['val'].append(cost(res1.x, X_val, y_val, 0))

plt.plot(errors['train'])
plt.plot(errors['val'])
plt.show()

#Create Polynomial method of degree 8
degree = 8
X_train_Poly = polynomialFeatures(X_train, degree)
mu = X_train_Poly.mean()
sigma = X_train_Poly.std()
X_train_Poly = (X_train_Poly - mu)/sigma

X_val_Poly = polynomialFeatures(X_val, degree)
X_val_Poly = (X_val_Poly - mu)/sigma

X_test_Poly = polynomialFeatures(X_test, degree)
X_test_Poly = (X_test_Poly - mu)/sigma

theta = np.zeros((degree+1,1))
res = optimize.minimize(cost, theta, args = (X_train_Poly, y_train, 0), method = 'CG', jac = gradientDescent, options = {'disp' :False})

errors = {'train':[], 'val':[]}
for i in range(len(X_train_Poly)):
    theta = np.zeros((degree+1,1))
    res1 = optimize.minimize(cost, theta, args = (X_train_Poly[:i+1], y_train[:i+1], 3), method = 'CG', jac = gradientDescent, options = {'disp' :False})
    errors['train'].append(cost(res1.x, X_train_Poly[:i+1], y_train[:i+1], 0))
    errors['val'].append(cost(res1.x, X_val_Poly, y_val, 0))

plt.plot(errors['train'])
plt.plot(errors['val'])
plt.show()

errors = {'train':[], 'val':[]}
lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
for lambda_ in lambda_vec:
    theta = np.zeros((degree+1,1))
    res1 = optimize.minimize(cost, theta, args = (X_train_Poly, y_train, lambda_), method = 'CG', jac = gradientDescent, options = {'disp' :False})
    errors['train'].append(cost(res1.x, X_train_Poly, y_train, 0))
    errors['val'].append(cost(res1.x, X_val_Poly, y_val, 0))

plt.plot(errors['train'])
plt.plot(errors['val'])
plt.show()