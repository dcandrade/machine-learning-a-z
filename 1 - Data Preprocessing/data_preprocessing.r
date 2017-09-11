# Importing the dataset
dataset = read.csv(file = "Data.csv")
#dataset = dataset[, 2:3] #

# Splitting the dataset in train and test set
library(caTools)
set.seed(123)
split = sample.split(Y = dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(x = dataset, subset = split == FALSE)

# Feature Scaling
#training_set[, 2:3] = scale(x = training_set[,2:3])
#testing_set[, 2:3] = scale(x = testing_set[,2:3])



