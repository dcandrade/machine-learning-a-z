# Importing the dataset
dataset = read.csv(file = "50_Startups.csv")
#dataset = dataset[, 2:3] #

dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1,2,3))

# Splitting the dataset in train and test set
library(caTools)
set.seed(123)
split = sample.split(Y = dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(x = dataset, subset = split == FALSE)

#Fitting Multiple Linear Regression model to the training set
regressor = lm(formula = Profit ~ ., data = training_set)

#Predicting test set results
y_pred = predict(regressor, newdata = testing_set)

#Optimal model using backward elimination (threshold = 0.5%)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)
