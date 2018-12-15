import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:,1:2].values
y = data.iloc[:,-1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)
y_pred = regressor.predict(X)

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X))
plt.show()

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid))
plt.show()
