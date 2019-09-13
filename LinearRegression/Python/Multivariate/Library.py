import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from datetime import datetime
from sklearn import linear_model

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
    
#Create Linear Regressor
lm = linear_model.LinearRegression(normalize = True)
startTime = datetime.now()
#Fit inputs
lm.fit(features, target)
print('Time taken: {}'.format( datetime.now() - startTime))

print('Paramters from Linear Regressor Model:[{},{}]'.format(lm.intercept_, lm.coef_))

#Predict Outputs
predicted_target = lm.predict(features)

#R2 score
print('R2_score:{}'.format(metrics.r2_score(target, predicted_target)))