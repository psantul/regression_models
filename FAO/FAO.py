import pandas as pd
import numpy as np

dataset= pd.read_csv('FAO.csv', encoding = "ISO-8859-1")
l = list([1])
for i in range(10,62):
    l.append(i)
X = dataset.iloc[:,l].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test = tts(X,y,test_size=0.2)

from sklearn.tree import DecisionTreeRegressor as d
r = d()
r = r.fit(X_train,y_train)

y_pred = r.predict(X_test)

loss = (np.sum(np.abs(y_pred-y_test))/np.sum(y_test))*100
acc = 100 - loss
