import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing datasets
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])
ohe = OneHotEncoder(categorical_features=[3])
X = ohe.fit_transform(X).toarray()

#Avoiding dummy variable trap
X = X[:, 1:]

#Splitting in training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test) #already fitted on training set

#Fitting Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting test set results
y_pred = regressor.predict(X_test)

#Building optimal model with backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones(shape = (50, 1)), values = X, axis = 1) #Considering first constant
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary() # Final Model