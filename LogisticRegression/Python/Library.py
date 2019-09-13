import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from datetime import datetime
from sklearn import linear_model

df = pd.read_csv('data/StudentGrades.csv', names = ['Exam1', 'Exam2', 'Admitted'])

df_pos = df[df['Admitted'] == 1]
df_neg = df[df['Admitted'] == 0]

ax = df_pos.plot(kind = 'scatter', x = 'Exam1', y = 'Exam2', c = 'green')
df_neg.plot(kind = 'scatter', x = 'Exam1', y = 'Exam2', c = 'red', ax = ax)
plt.title('Grades vs Admission Status')
plt.show()

features = df[df.columns[0:-1]]
target = df[df.columns[-1]]

#Create Linear Regressor
lm = linear_model.LogisticRegression()
startTime = datetime.now()

#Fit inputs
lm.fit(features, target)
print('Time taken: {}'.format( datetime.now() - startTime))

print('Paramters from Linear Regressor Model:[{},{}]'.format(lm.intercept_, lm.coef_))

#Predict Outputs
predicted_target = lm.predict(features)
    
#R2 score
print('R2_score:{}'.format(metrics.r2_score(target, predicted_target)))

cf_matrix = metrics.confusion_matrix(target, predicted_target)

print('F1 score:{}'.format(metrics.f1_score(target, predicted_target)))