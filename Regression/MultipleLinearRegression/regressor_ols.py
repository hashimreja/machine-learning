import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Startups.csv')

X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,3] = le.fit_transform(X[:,3])
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = [3])
X = ohe.fit_transform(X).toarray()

X = X[:,1:]
#avoiding dummy variable trap

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/5)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm

X = np.append(arr=np.ones((50,1)).astype(int),values = X,axis =1)

X_optimal = X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = X_optimal).fit()
regressor_ols.summary()


X_optimal = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = X_optimal).fit()
regressor_ols.summary()


X_optimal = X[:,[0,3]]
regressor_ols = sm.OLS(endog = y,exog = X_optimal).fit()
regressor_ols.summary()
