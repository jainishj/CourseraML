import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('data/ex2data1.txt', header = None)

X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values.reshape(-1,1)

plt.figure(0)
plt.scatter(X[:,0], X[:,1], c = y[:,0])
plt.xlabel('Score 1')
plt.ylabel('Score 2')
plt.title('Admission vs Scores')
plt.show()

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

feature_set = np.hstack((np.ones([X.shape[0],1]), X))

#Fitting Logistic Regresison to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(feature_set, y)

y_pred = classifier.predict(feature_set)

ans = y_pred.reshape(-1,1) == y
print("Theta:", classifier.coef_)