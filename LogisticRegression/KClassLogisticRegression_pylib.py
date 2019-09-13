import pandas as pd
import numpy as np

pixel_data = pd.read_csv('data/pixel_data.csv', header = None)
y_data = pd.read_csv('data/pixel_data_results.csv', header = None)

input_ = pixel_data.values
output_ = y_data.values.ravel()
training_size = input_.shape[0]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
input_ = sc_X.fit_transform(input_)

feature_set = np.hstack((np.ones([input_.shape[0],1]), input_))

#Fitting Logistic Regresison to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class = 'multinomial' ,random_state = 0, solver='lbfgs')
classifier.fit(feature_set, output_)

y_pred = classifier.predict(feature_set)

quality = (output_.reshape(-1,1) == (y_pred.reshape(-1,1)))
print (np.count_nonzero(quality == True)/training_size*100)