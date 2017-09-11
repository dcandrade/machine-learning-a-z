# Preprocessing data using Pandas dataframes as main structure
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Remove false-positive SettingWithCopyWarning from pandas
#pd.options.mode.chained_assignment = None

#Importing datasets
dataset = pd.read_csv("Data.csv")
X = dataset[['Country', 'Age', 'Salary']]
y = dataset['Purchased']

#Handling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
incomplete_columns = ['Age', 'Salary']
imputer = imputer.fit(X[incomplete_columns]) #Fitting columns with missing values
X[incomplete_columns] = imputer.transform(X[incomplete_columns]) #Replace missing data on X

#Encoding categorical variables
X = pd.get_dummies(X, columns=['Country'])

#Splitting in training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #already fitted on training set