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
import patsy
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('dane_slimak.csv', encoding = 'utf-8', delim_whitespace=True)
# sprawdzenie, ze nie ma nullowych wartosci w zbiorze danych
df.isnull()
#slimaki_df.isnull().sum()
#print(df.isnull().sum())
df['Sex'].replace({'I': 2, 'F': 1, 'M': 0},inplace = True)
#print(df.head())
#Macierz korelacji - wstepna propozycja to niepowtarzanie feature engineering Length i Diameter, bo są mocno skorelowane i mają podobną korelację
sns.heatmap(df.corr(), cmap = 'seismic', annot=True, fmt=".2f")
plt.show()
df['age']= (df['Rings']+(1.5)).astype(float)
X = df.drop(['Rings'], axis = 1)
y = df['Rings']

# print(df)
# print(df.head())
# print(df.groupby('Sex').mean())
f = 'Rings~Shell_weight*Height'
y, X = patsy.dmatrices(f,df, return_type="dataframe")
print(patsy.dmatrices)
#print(X.columns)

y = np.ravel(y)
# # instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)
#
# # check the accuracy on the training set
model.score(X, y)
print('model score',model.score(X, y))

pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
print(pd.DataFrame(zip(X.columns, np.transpose(model.coef_))))
