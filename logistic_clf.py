#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split


n_samples = 10000

centers = [(-1, -1), (1, 1)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=42)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
log_reg = linear_model.LogisticRegression()

# some code here
# fit and predict
log_reg.fit(X_train,y_train)
predict = log_reg.predict(X_test)
wrongcount = 0

for i in range(0,predict.shape[0]):
    # check if there are any invalid predictions
    if((predict[i]!=0) and (predict[i]!=1)):
        print("Yes")
        break

    # get # wrong predictions
    if(predict[i] != y_test[i]):
        wrongcount += 1

# print # wrong predictions
print("Number of wrong predictions is: ", wrongcount)

# scatter plot the data with classes
plt.scatter(X_test[:,0],X_test[:,1], c=predict)
plt.xlabel("X_test[0]")
plt.ylabel("X_test[1]")
plt.title("Classification with Logistic Regression")
plt.show()
quit()
