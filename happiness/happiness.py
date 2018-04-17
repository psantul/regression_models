#importing libraries
import pandas as pd
import numpy as np

#importing and preparing dataset
dataset = pd.read_csv('2015.csv')
l=list()
l.append(1)
for i in range(10,12):
    l.append(i)
X = dataset.iloc[:, l].values
y = dataset.iloc[:, 3].values

#Catagorical data handling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[: , 0].astype(str))

#spiliting dataset
from sklearn.model_selection import train_test_split as tts
X_train,X_test,y_train,y_test = tts(X,y,test_size=0.2,random_state=0)

#training the regressor
from sklearn.linear_model import LinearRegression as lr
regressor = lr()
regressor = regressor.fit(X_train,y_train)

#predicting the result
y_pred = regressor.predict(X_test)

#calculating the loss and accuracy of result
loss = (np.sum(np.abs(y_pred-y_test))/np.sum(y_test))*100
acc = 100-loss
