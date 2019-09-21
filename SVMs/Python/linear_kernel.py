import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
import numpy as np

random_df = pd.read_csv('data/Random1.csv')

X = random_df.iloc[:,:-1]
y = random_df.iloc[:,-1]

pos_val = X[y == 1]
neg_val = X[y == 0]

fig, ax = plt.subplots()
ax.plot(pos_val.iloc[:,0].values, pos_val.iloc[:,-1].values, 'b+')
ax.plot(neg_val.iloc[:,0].values, neg_val.iloc[:,-1].values, 'go')

clf = SVC(C =1, kernel = 'linear')
clf.fit(X, y)

y_pred = clf.predict(X)

print('Accuracy:{}, F1 score:{}'.format(accuracy_score(y, y_pred), f1_score(y, y_pred)))

w = clf.coef_[0]
b = clf.intercept_;
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)

yp = - (w[0]*xx + b)/w[1];
ax.plot(xx, yp)

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
          linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')
plt.show()

Cs = np.linspace(0.1,0.5,20)
grid = GridSearchCV(estimator=clf, param_grid=dict(C=Cs))
grid.fit(X, y)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.C)