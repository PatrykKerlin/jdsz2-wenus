import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

import patsy
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('dane_slimak.csv', encoding = 'utf-8', delim_whitespace=True)
train_X, test_X  = train_test_split(df, random_state=101)


###---------- interakcje ----------------------
f = 'Rings~Whole_weight*Length'
y, X = patsy.dmatrices(f,train_X, return_type="dataframe")
y_t, X_t = patsy.dmatrices(f,test_X, return_type="dataframe")

y = np.ravel(y)
y_t = np.ravel(y_t)

# # instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)

# # check the accuracy on the training set
pr = model.predict(X_t)

errors = (abs(pr - y_t)/y_t)*100
print('MAPE with interactions:',np.mean(errors))
print(sm.OLS(y, X).fit().summary())
##------------- bez interakcji --------------------------

f = 'Rings~Whole_weight+Length'
y, X = patsy.dmatrices(f,train_X, return_type="dataframe")
y_t, X_t = patsy.dmatrices(f,test_X, return_type="dataframe")

y = np.ravel(y)
y_t = np.ravel(y_t)

# # instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)
#
# # check the accuracy on the training set
pr = model.predict(X_t)

errors = (abs(pr - y_t)/y_t)*100
print('MAPE without interactions:',np.mean(errors))
print(sm.OLS(y, X).fit().summary())
#WYNIKI LogisticRegression
# f = 'Rings~Shell_weight*Height' MAPE: 17.4472
# f = 'Rings~Shell_weight+Height' MAPE: 17.4952

# f = 'Rings~Diameter*Whole_weight' MAPE: 16.9314
# f = 'Rings~Diameter+Whole_weight' MAPE: 16.9696

# f = 'Rings~Diameter*Viscera_weight' MAPE:  17.4893
# f = 'Rings~Diameter+Viscera_weight' MAPE: 17.5508

# f = 'Rings~Whole_weight*Length' MAPE: 16.7179
#f = 'Rings~Whole_weight*Length' MAPE:  16.7524
