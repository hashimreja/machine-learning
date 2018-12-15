import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

#preprocessing using scaling

from sklearn.preprocessing import StandardScaler
sc_x =StandardScaler()
sc_y =StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

#fitting regressor

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)
#prediction

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

#visualisation
plt.scatter(X,y,color='green')
plt.plot(X,regressor.predict(X),color='red')
plt.title('bluffer detector')
plt.Xlabel('experience')
plt.ylabel('salary')
plt.show()

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color ='pink')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('bluffer detector')
plt.Xlabel('experience')
plt.ylabel('salary')
plt.show()


