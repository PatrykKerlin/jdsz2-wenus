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


df = pd.read_csv('dane_slimak.csv', encoding = 'utf-8', delim_whitespace=True)
df['Sex'].replace({'I': 2, 'F': 1, 'M': 0},inplace = True)

# #Macierz korelacji - wstepna propozycja to niepowtarzanie feature engineering Length i Diameter, bo są mocno skorelowane i mają podobną korelację
# sns.heatmap(df.corr(), cmap = 'seismic', annot=True, fmt=".2f")
# plt.show()

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
print(df.head())
f = 'Rings ~ Diameter + Sex * Diameter'
y_train, X_train = patsy.dmatrices(f, df, return_type='dataframe')
X_train
print(patsy.dmatrices)
# Linear regression
# clf_linear = LinearRegression()
# clf_linear.fit(X_train, y_train)
# accuracy_linear = clf_linear.score(X_test, y_test)
# y_pred_linear = clf_linear.predict(X_test)
# mae_linear = mean_absolute_error(y_test, y_pred_linear)
#
# print('Accuracy linear:', accuracy_linear)
# print('MAE Linear regression:', mae_linear)
