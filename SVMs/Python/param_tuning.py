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
ax.plot(pos_val[:,0], pos_val[:,1], 'go')
ax.plot(neg_train[:,0], neg_train[:,1], 'ro')
ax.plot(neg_val[:,0], neg_val[:,1], 'ro')
plt.show()

C = np.linspace(0.1,10,20)
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