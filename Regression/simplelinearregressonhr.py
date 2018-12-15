import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3 ,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
#visualizing the training set
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('training set')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()
#visualizing the testing set
plt.scatter(X_test,y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('testing set')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()
