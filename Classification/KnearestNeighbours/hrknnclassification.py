import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Social_Network_Ads.csv')

X = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X = sc_X.fit_transform(X)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(X_train,y_train)

y_pred = kn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from matplotlib.colors import ListedColormap
#visualizing training set
x_set,y_set = X_train,y_train
x1 , x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1,stop = x_set[:,0].max() + 1,step = 0.01),
                      np.arange(start = x_set[:,1].min() - 1,stop = x_set[:,1].max() + 1,step = 0.01),
                      )
plt.contourf(x1, x2 ,kn.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75,
             cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i , j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1],c =
                ListedColormap(('red', 'green'))(i),label = j)
plt.title('knn neighbors classifier')
plt.xlabel('age')
plt.ylabel('salary')
plt.legend()
plt.show()

