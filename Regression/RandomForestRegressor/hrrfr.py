import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:,1:2].values
y = data.iloc[:,-1].values

#regressor
from sklearn.ensemble import RandomForestRegressor
#n_estimators means no trees you want to built 
regressor = RandomForestRegressor(n_estimators = 1000)
regressor.fit(X,y)
y_pred = regressor.predict(X)

#visualizing the data
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color ='yellow')
plt.plot(X_grid,regressor.predict(X_grid),color='red')
plt.title('bluff detector')
plt.xlabel('experience')
plt.ylabel('salary')
plt.legend()
plt.show()
