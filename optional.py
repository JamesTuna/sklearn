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

# iterate all degrees,
# and get the maximum prediction degreee
pr_score=[]
ridge_score=[]
for i  in range(0,20):
    poly = PolynomialFeatures(degree=i)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # build a linear regression model
    # and record the scores
    pr_model = LinearRegression()
    pr_model.fit(X_train_poly,y_train)
    pr_score.append(pr_model.score(X_test_poly,y_test))

    # build a linear regression model
    # and record the scores
    ridge_model = Ridge(alpha=1, normalize=False)
    ridge_model.fit(X_train_poly, y_train)
    ridge_score.append(ridge_model.score(X_test_poly,y_test))
'''
print("pr_score")
print(pr_score)
print("")
print("ridge_score")
print(ridge_score)
print("")
'''
# print the maximum degreee of score
print("max pr score: ", max(pr_score), "degree: ",pr_score.index(max(pr_score))+1)
print("max ridge score: ", max(ridge_score), "degree: ",pr_score.index(max(pr_score))+1)
