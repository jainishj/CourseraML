import pandas as pd
import numpy as np
#Read Data
df = pd.read_csv('data/Numbers_Pixel.csv')

#Extract features and target from dataframe
features = df[df.columns[0:-1]]
target = df.iloc[:,-1]

#Fitting Logistic Regresison to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class = 'multinomial' ,random_state = 0, solver='lbfgs')
classifier.fit(features, target)

y_pred = classifier.predict(features)

quality = (target == y_pred)
print (np.count_nonzero(quality == True)/len(features)*100)