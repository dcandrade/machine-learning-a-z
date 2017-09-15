# Importing the dataset
dataset = read.csv(file = "Salary_Data.csv")
#dataset = dataset[, 2:3] #

# Splitting the dataset in train and test set
library(caTools)
set.seed(123)
split = sample.split(Y = dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(x = dataset, subset = split == FALSE)

# Fitting linear regression model to the training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# Predicting test set results
y_pred = predict(regressor, newdata = testing_set)

# Visualizing training set results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y =  predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary x  Experience - Training set') +
  xlab('Years of Experience') + 
  ylab('Annual Salary')

# Visualizing testing set results
ggplot() + 
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y =  predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary x  Experience - Testing set') +
  xlab('Years of Experience') + 
  ylab('Annual Salary')