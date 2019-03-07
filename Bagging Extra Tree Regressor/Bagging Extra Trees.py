# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:34:05 2019

@author: SYED HASHIM REJA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

d1 = pd.read_csv('features.csv')
data = pd.read_csv('Machine.csv',names = d1)

corr = data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(data = corr,annot = True)

X = data.iloc[:,3:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#bagging model
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
regressor = ExtraTreesRegressor()
model = BaggingRegressor(base_estimator = regressor,n_estimators =150)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#metrics for evaluating the model
from sklearn import metrics
metrics.mean_absolute_error(y_test,y_pred)
#r2 score metric
model.score(X_train,y_train)
model.score(X_test,y_test)

#cross_val_score
'''
from sklearn.model_selection import KFold
kfold = KFold(n_splits = 10,shuffle= True)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor(base_estimator = regressor,n_estimators =100)
score = cross_val_score(model,X_train,y_train,cv = kfold,scoring = 'mean_absolute_error')
print('score_mean{}'.format(score.mean()))
'''













