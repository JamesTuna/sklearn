import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv
import seaborn
seaborn.set()

df = pd.read_csv('imports-85.data',
            header=None,
            names=['symboling','normalized-losses','make','fuel-type',
                'aspiration','num-of-doors','body-style','drive-wheels',
                'engine-location','wheel-base','length','width','height',
                'curb-weight','engine-type','num-of-cylinders','engine-size',
                'fuel-system','bore','stroke','compression-ratio','horsepower',
                'peak-rpm','city-mpg','highway-mpg','price'],
                na_values=('?'))
print(df.shape)

# remove NaN values
df = df.dropna()
print(df.shape)

# get feature matrix
feature_matrix = df.as_matrix(columns=['horsepower','engine-size','peak-rpm'])

# standardize the data
transformer = StandardScaler()
X = transformer.fit_transform(feature_matrix)
y = transformer.fit_transform(df['price'].values.reshape(-1,1))

# Matrix manipulation
new_X = np.ones((X.shape[0],X.shape[1]+1))
new_X[:,1:]=X
print(new_X.shape);
theta = np.dot(np.transpose(new_X),new_X)
theta = inv(theta)
theta = np.dot(theta, np.transpose(new_X))
theta = np.dot(theta, y)
print("Parameter theta calculated by normal equation: ", theta)

# Solve multiple linear regression with gradient descent
clf = linear_model.SGDRegressor(loss='squared_loss')
clf.fit(X, y[:,0])
print("Parameter theta calculated by SGD:", clf.intercept_, clf.coef_)


quit()
