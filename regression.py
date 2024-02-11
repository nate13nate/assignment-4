# Simple Linear Regression
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Importing the datasets
datasets = pd.read_csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
train, test = train_test_split(datasets)

# Fitting Simple Linear Regression to the training set
regressor = LinearRegression()
regressor.fit(train['YearsExperience'].to_frame(), train['Salary'].to_frame())

# Predicting the Test set result ￼
yTrainPred = regressor.predict(train['YearsExperience'].to_frame())
yPred = regressor.predict(test['YearsExperience'].to_frame())

# Get mean squared error
error = mean_squared_error(test['Salary'].array, yPred)
print(f'MEAN SQUARED ERROR: {error}')

# Visualising the Training set results
ax = train.plot.scatter(x = 'YearsExperience', y = 'Salary', c = 'blue')

modelTrainRes = pd.concat([train.reset_index()['YearsExperience'], pd.Series((x[0] for x in list(yTrainPred)), name='Salary')], axis=1)
modelTrainRes.plot(ax = ax, x = 'YearsExperience', y = 'Salary', title = 'Training Data vs Regression')
plt.legend(['Training Data', 'Regression'])

# Visualising the Test set results
ax2 = test.plot.scatter(x = 'YearsExperience', y = 'Salary', c = 'red')

modelTestRes = pd.concat([test.reset_index()['YearsExperience'], pd.Series((x[0] for x in list(yPred)), name='Salary')], axis=1)
modelTestRes.plot(ax = ax2, x = 'YearsExperience', y = 'Salary', title = 'Test Data vs Regression')
plt.legend(['Test Data', 'Regression'])
plt.show()

