import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from datetime import datetime

#Read data
df = pd.read_csv('data/HousingPrice.csv', names = ['Size', 'Bedrooms', 'Price'])

#Extract features
features = df[df.columns[0:-1]]

#Extract Target
target = df[df.columns[-1]]
target_name = target.name

for i in range(len(features.columns)):
    col_name = features.columns[i]
    df.plot(kind = 'scatter', x = col_name, y = target_name)
    plt.title('{} vs {}'.format(col_name, target_name))
    plt.show()


startTime = datetime.now()

#Normal Equation
X = np.insert(features.values, 0, np.ones(len(features),), axis = 1)
theta = np.linalg.pinv(X.T@X)@X.T@target
print('Time taken: {}'.format( datetime.now() - startTime))

print ('Theta calculated from normal equation: {}'.format(theta.tolist()))

predicted_target = np.insert(features.values, 0, np.ones(len(features),), axis = 1)@theta.T
    
#R2 score
print('R2_score:{}'.format(metrics.r2_score(target, predicted_target)))