#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

X_train = [[5.3], [7.2], [10.5], [14.7], [18], [20]]
y_train = [[7.5], [9.1], [13.2], [17.5], [19.3], [19.5]]

X_test = [[6], [8], [11], [22]]
y_test = [[8.3], [12.5], [15.4], [19.6]]

lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
score = lr_model.score(X_test,y_test)
print("Linear regression score is: ",score)

poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# some code here
# build a linear regression model
pr_model = LinearRegression()
pr_model.fit(X_train_poly,y_train)
print('--------------------')
print(pr_model.intercept_, pr_model.coef_)
score = pr_model.score(X_test_poly,y_test)
print("Linear regression (order 5) score is: ",score)

xx = np.linspace(0, 26, 100)
xx_poly = poly.transform(xx.reshape(xx.shape[0], 1))
yy_poly = pr_model.predict(xx_poly)

# some code here
# plot
plt.plot(xx, yy_poly,'r', X_test, y_test, 'b*')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear regression (order 5) result")
plt.show()

# some code here
ridge_model = Ridge(alpha=1, normalize=False)
ridge_model.fit(X_train_poly, y_train)

# get coef_ and intercept_
# predict the score
print('--------------------')
print(ridge_model.intercept_, ridge_model.coef_)
score = ridge_model.score(X_test_poly,y_test)
print("Ridge regression (order 5) score is: ",score)

# plot
yy_ridge = ridge_model.predict(xx_poly)
plt.plot(xx, yy_ridge,'r', X_test, y_test, 'b*')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ridge regression (order 5) result")
plt.show()
