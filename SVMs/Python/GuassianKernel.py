import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
import numpy as np

def guassian_kernel(x1, x2):
    sigma = 0.1
    return np.exp(-((x1-x2)**2)/(2*(sigma**2)))

random_df = pd.read_csv('data/Random2.csv')

X = random_df.iloc[:,:-1].values
y = random_df.iloc[:,-1].values

pos_val = X[y == 1]
neg_val = X[y == 0]

fig, ax = plt.subplots()
ax.plot(pos_val[:,0], pos_val[:,-1], 'b+')
ax.plot(neg_val[:,0], neg_val[:,-1], 'go')

def my_kernel(X1, X2):
   cc = np.zeros((len(X1),len(X2)))
   for i in range(len(X1)):
    for j in range(len(X2)):
        cc[i][j] = guassian_kernel(X1[i,0], X2[j,1])
        
    return cc


clf = SVC(C =.1, kernel = my_kernel, gamma = 'auto')
clf.fit(X, y)

y_pred = clf.predict(X)

print('Accuracy:{}, F1 score:{}'.format(accuracy_score(y, y_pred), f1_score(y, y_pred)))

Cs = np.linspace(0.1,0.5,20)
grid = GridSearchCV(estimator=clf, param_grid=dict(C=Cs))
grid.fit(X, y)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.C)