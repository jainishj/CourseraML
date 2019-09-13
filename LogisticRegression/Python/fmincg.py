import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import metrics

def sigmoid(val):
    return 1/(1+np.exp(-val))

def cost(theta, *args):
    X, y = args
    cost = (-(y.transpose()@np.log(sigmoid(X@theta))) - ((1-y).transpose()@np.log(1-sigmoid(X@theta))))/X.shape[0];
    return cost[0]

def grad(theta, *args):
    theta = theta.reshape(-1,1)
    X, y = args
    gg = (X.transpose()@(sigmoid(X@theta) - y))/X.shape[0]
    return gg.flatten()

#Read Data
df = pd.read_csv('data/StudentGrades.csv', names = ['Exam1', 'Exam2', 'Admitted'])

#Split Positive & Negative Data
df_pos = df[df['Admitted'] == 1]
df_neg = df[df['Admitted'] == 0]

#Plot
ax = df_pos.plot(kind = 'scatter', x = 'Exam1', y = 'Exam2', c = 'green')
df_neg.plot(kind = 'scatter', x = 'Exam1', y = 'Exam2', c = 'red', ax = ax)
plt.title('Grades vs Admission Status')
plt.show()

#Extract features and target from dataframe
features = df[df.columns[0:-1]]
target = df[df.columns[-1]]

features_with_bias = np.insert(features.values, 0, np.ones(len(features),), axis = 1)

theta = np.array([0.0]*(len(features.columns)+1)).reshape(-1,1)
  
startTime = datetime.now()

#Call fmin cg function
res1= optimize.fmin_cg(cost, theta, fprime=grad, args=(features_with_bias, target.values.reshape(-1,1)))

print('Time taken: {}'.format( datetime.now() - startTime))

#Predictions on parameters
predicted_target = np.round(sigmoid(features_with_bias@res1))

print ('Theta after {} fmincg'.format(res1.tolist()))

#R2 score
print('R2_score:{}'.format(metrics.r2_score(target, predicted_target)))

cf_matrix = metrics.confusion_matrix(target, predicted_target)

print('F1 score:{}'.format(metrics.f1_score(target, predicted_target)))