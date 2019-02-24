#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:23:48 2019

@author: nanokoper
"""

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
import patsy

path = '/home/nanokoper/Pulpit/ISA/jdsz2-wenus/projekt_ML/dane_slimak.csv'
df = pd.read_csv(path, encoding = 'utf-8', delim_whitespace=True)
df['Sex'].replace({'I': 2, 'F': 1, 'M': 0},inplace = True)

#Macierz korelacji - wstepna propozycja to niepowtarzanie feature engineering Length i Diameter, bo są mocno skorelowane i mają podobną korelację
sns.heatmap(df.corr(), cmap = 'seismic', annot=True, fmt=".2f")
plt.show()

X = df.drop(['Rings'], axis = 1)
y = df['Rings']

#wykresy dla parametrow w modelach
#funkcja kosztu

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 100)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state = 100)

"""normalizacja z outleiersami
norm_scale = preprocessing.StandardScaler().fit(df[['Sex', 'Height']])
df_norm = norm_scale.transform(df[['Sex', 'Height']])plt.figure(figsize=(10,10))
plt.scatter(df_norm[:,0], df_norm[:,1], color='blue', alpha=0.3)"""

#Linear regression
clf_linear = LinearRegression()
clf_linear.fit(X_train, y_train)
accuracy_linear = clf_linear.score(X_test, y_test)
y_pred_linear = clf_linear.predict(X_test)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

#SVR
clf_SVR = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf_SVR.fit(X_train, y_train)
accuracy_SVR = clf_SVR.score(X_test, y_test)
y_pred_SVR = clf_SVR.predict(X_test)
mae_SVR = mean_absolute_error(y_test, y_pred_SVR)

#randomforestregressor
clf_RFC = RandomForestRegressor()
clf_RFC.fit(X_train, y_train)
accuracy_RFC = clf_RFC.score(X_test, y_test)
y_pred_RFC = clf_RFC.predict(X_test)
mae_RFC = mean_absolute_error(y_test, y_pred_RFC)

#xgboostregressor
clf_XGB = xg.XGBRegressor()
clf_XGB.fit(X_train, y_train)
accuracy_XGB = clf_XGB.score(X_test, y_test)
y_pred_XGB = clf_XGB.predict(X_test)
mae_XGB = mean_absolute_error(y_test, y_pred_XGB)

#DecisionTreeRegressor
clf_DTR = DecisionTreeRegressor()
clf_DTR.fit(X_train, y_train)
accuracy_DTR = clf_DTR.score(X_test, y_test)
y_pred_DTR = clf_DTR.predict(X_test)
mae_DTR = mean_absolute_error(y_test, y_pred_DTR)

print('Accuracy linear:', accuracy_linear)
print('Accuracy SVR:', accuracy_SVR)
print('Accuracy RFC:', accuracy_RFC)
print('Accuracy XGB:', accuracy_XGB)
print('Accuracy DTR:', accuracy_DTR)

print('MAE Linear regression:', mae_linear)
print('MAE SVR:', mae_SVR)
print('MAE RFC:', mae_RFC)
print('MAE XBG:', mae_XGB)
print('MAE DTR:', mae_DTR)

""" To do: patsy, crossvalidacja, (model.summary(), OLS, coefficient matrix i inne takie, MSE) - inne funkcje kosztu i sprawdzenie modeli, 
dopasowanie parametrów modeli. Dostosowanie kodu do kaggle - nie tykamy testowych danych"""
