#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading file
data = pd.read_csv('data.csv')
data.head()
X = data.iloc[:,:-1].values
y = data.iloc[:,3].values
#using imputer to handle missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',axis =0,strategy = 'mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
#using labelencoder to transform str to int and onehotencoder to create dummy variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features =[0])
X = onehotencoder.fit_transform(X)
#encoding the y with labelencoder
y = labelencoder.fit_transform(y)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)