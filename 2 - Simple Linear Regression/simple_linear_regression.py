import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing datasets
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting in training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#Feature Scaling is done while fitting the model
#==============================================================================
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test) #already fitted on training set
#==============================================================================

#Fitting model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting test results
y_pred = regressor.predict(X_test)

#Visualizing the results - Training Set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary x Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Annual Salary')
plt.show()

#Visualizing the results - Testing Set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary x Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Annual Salary')
plt.show()