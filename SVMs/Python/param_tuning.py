import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('data/training.csv')
df_val = pd.read_csv('data/validation.csv')

X_train = df_train.iloc[:,:-1].values
y_train = df_train.iloc[:,-1].values

X_val = df_val.iloc[:,:-1].values
y_val = df_val.iloc[:,-1].values

pos_train = X_train[y_train == 1]
neg_train = X_train[y_train == 0]

pos_val = X_val[y_val == 1]
neg_val = X_val[y_val == 0]

fig, ax = plt.subplots()
ax.plot(pos_train[:,0], pos_train[:,1], 'go')
ax.plot(neg_train[:,0], neg_train[:,1], 'ro')

C = np.linspace(0.1,100,200)
max_acc_score = 0
best_C = 0
for c in C:
    clf = SVC(C = c, kernel = 'rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc_score = accuracy_score(y_pred, y_val)
    print(acc_score)
    if(acc_score > max_acc_score):
        max_acc_score = acc_score
        best_C = c
        
print('Best C:{} with accuracy score:{}'.format(best_C, max_acc_score))

clf = SVC(C =c, kernel = 'rbf')
clf.fit(X_train, y_train)

fig, ax1 = plt.subplots()
ax1.plot(pos_val[:,0], pos_val[:,1], 'go')
ax1.plot(neg_val[:,0], neg_val[:,1], 'ro')

# create grid to evaluate model
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
# plot decision boundary and margins
ax1.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
          linestyles=['--', '-', '--'])
# plot support vectors
ax1.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')
plt.show()
