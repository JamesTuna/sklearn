import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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

# split data into training and testing
split = round(df.shape[0] * 0.8);
df_train = df[:split]
df_test = df[split:]
print('Shape of train data is'+str(df_train.shape))
print('Shape of test data is'+str(df_test.shape))

# data standarzation
transformer = StandardScaler()
scaled_train_horsepower = transformer.fit_transform(df_train['horsepower'].values.reshape(-1,1))
scaled_test_horsepower = transformer.transform(df_test['horsepower'].values.reshape(-1,1))
scaled_train_price = transformer.fit_transform(df_train['price'].values.reshape(-1,1))
scaled_test_price = transformer.transform(df_test['price'].values.reshape(-1,1))

# build a linear regression model
model = LinearRegression()
model.fit(scaled_train_horsepower,scaled_train_price)
predict = model.predict(scaled_test_horsepower)

# plot
plt.scatter(scaled_test_horsepower, predict, color='red')
plt.scatter(scaled_test_horsepower, scaled_test_price, color='blue')
plt.plot(scaled_test_horsepower, predict, color='black', linewidth=2)
plt.xlabel("Standardized horsepower")
plt.ylabel("Standardized Price")
plt.title("Linear regression on cleaned and standardized data")
plt.savefig('lr_rawdata.png')
plt.show()
quit()
