+++
title = "Machine learning project"
date = "2018-12-11"
tags = ["random1", "random2"]
categories = ["Category 11"]
description = "A brief summary for the post 2"
+++

# Introduction

It is time to speak about our first Machine Learning project. Me and my friends are team mates at post-operative life expectancy of lung cancer project, we try to work on lung cancer as it is considered as one of the most common forms of cancer in today’s world. Thoracic surgery is one of the ways to diagnose lung cancer if it is detected at an early stage. Hence it is better to cure lung cancer at the beginning stage. Patients survival cannot be predicted by the surgery alone. Hence if the patient’s survival cannot be extended for a year after surgery, then the factors for the death remains a mystery so in order to overcome this problem we are here used data mining techniques in our project to detect the patient’s survival. This will help in taking the decision to enter the surgery or not. 

# Data Pre-processing and visualization:
First of all, we download our data from UCI machine learning. 

http://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data

```
library(readxl)
dataset <- read_excel("C:/Users/pc/Desktop/thoracic.xlsx")
View(dataset)

```
We know that we have to get rid of outliers for better machine learning model as they affect mean and standard deviation. For the previous reason, we visualize dataset with age feature so we find that the people with 21 years old are outlier. Also, we visualize PRE5 with data (PRE5: Volume that has been exhaled at the end of the first second of forced expiration - FEV1) and there were 15 outliers so we eliminate the rows of them. 

```
plot(dataset$AGE)

#from the age one outlier-- we need to remove this outlier (age = 21)

subset(dataset, AGE==21 )

dataset <-dataset[-c(397),] 
```
```
plot(dataset$PRE5)
# we have outliers between 40 and 100 in the pre5 col.
plot(dataset$PRE5)
subset(dataset, PRE5>7 )
dataset <-dataset[-c(26,90,99,113,133,216, 256,320,326, 331,350, 353,354,439,445),] 
```
Why have we name everything? To make it more readable and that’s why we rename our features name. 
```
library(caTools)
library(tidyverse)
dataset<- dataset%>%  rename( Diagnosis= DGN  ,FVC = PRE4 , FEV1 = PRE5,Pef_stat= PRE6 , Pain_surg = PRE7,
                              Haemoptysis_bf_surgery=PRE8, Dyspnoea_bef_surg= PRE9, Cough_bef_surg = PRE10,
                              Weakness_bef_surg= PRE11,size_original_tumour=PRE14,Type2DM= PRE17, MI =PRE19,
                              PAD=  PRE25 , smoking= PRE30 , Asthma= PRE32)
```
when you will explore your data, you will find id column that was useless so eliminate it 
```
### remove the id col.
dataset <- subset(dataset, select = -c(id))
```
Then you will split data to train set and test set to the ratio of 80% training and 20% test
```
library(caret)


### data spliting

set.seed(100) # for the reproducability of data randomly on each time you run the code

# Step 1: Get row numbers for the training data
trainRowNumbers <- createDataPartition(dataset$Risk1Yr, p=0.8, list=FALSE)

# Step 2: Create the training  dataset
train <- dataset[trainRowNumbers,]

# Step 3: Create the test dataset
test <- dataset[-trainRowNumbers,]
```
feature scaling is done to make all the values in each feature from 0 and 1

```
### feture scalling
preProcess_range_model <- preProcess(train, method='range')
train <- predict(preProcess_range_model, newdata = train)
test <- predict(preProcess_range_model, newdata = test)
```
###Take away: 
There are some libraries are required to be installed so take care my friend 
Fortunately, we don’t have any missing data so that’s our pre-processing steps. Explore your data carefully

### Data visualization:
![visualization of each feature](C:\Users\pc\ga_blog\_posts\1.jpg)
 

![decesion tree RMS](C:\Users\pc\ga_blog\_posts\image.jpg)

For the second method we used the decision tree method to find out if its result will be better than linear regression method or not, but we faced the same problem because of unbalanced data so we finally decided to use linear regression with Bagging to solve this problem.

# Code Implementation

So far we have been talking about using machine learning to detect the survivng rate, but how can we implement such a thing?
By using R we can write a code that is able to do this task by understanding the used dataset we caan conclude that the problem is a classification problem. According to machine learning techniques a classification problem requires certain methods.

Some of the methods we have used are, logistic regression and dissision trees.

## Logistic regression

One of the models used to handle a classification problem with two possible outcomes 'True or False' is a logistic regression model. But how to implement it in R ?
It is so simple, by using a backage in R called caret you can implement a logistic regression model as the following: 
```
library(caret)
set.seed(123)
#the following line is for a 10 fold cross validation 
train.control <- trainControl(method = "cv", number = 10)
train$Risk1Yr = as.factor(train$Risk1Yr)

trainnew$Risk1Yr = as.factor(trainnew$Risk1Yr)
#this is how to impelement a logistic regression model , just use glm !!
model <- train(Risk1Yr ~ .,
               data = trainnew,
               trControl = train.control,
               method = "glm",
               family=binomial())

print(model)
#confusion matrix to detect the accuracy of the model
pred<- predict(model, test[,1:16])
acc<- confusionMatrix(data= pred0,reference=test$Risk1Yr,mode='everything', positive= '1')
acc
```
While implimenting this model we got a very unaccurate results, the main reason of that was the fact that we use an imbalanced data which means that one class of the data 'False' is the majority of our samples with a very small amount of 'True'.

We fixed this problem by using a down sampling technique, this technique qualizes the two clases in the data.

```
trainnew <- downSample(train[,1:16] ,train$Risk1Yr)
```
By using this new training set the result improved alot.

## Decision trees

It is another way to train a classification model, by using an rpart backage the impelimentaion is

```
library(foreach)
###tree bagging
#Create a parallel socket cluster
cl <- makeCluster(8) # use 8 workers
registerDoParallel(cl) # register the parallel backend

#Fit trees in parallel and compute predictions on the test set
predictions <- foreach(
  icount(160), 
  .packages = "rpart", 
  .combine = cbind
) %dopar% {
  #bootstrap copy of training data
  index <- sample(nrow(train), replace = TRUE)
  train_boot <- train[index, ]  
  
  #fit tree to bootstrap copy
  bagged_tree <- rpart(
    Risk1Yr ~ ., 
    control = rpart.control(minsplit = 2, cp = 0),
    data = train_boot
  ) 
  
  predict(bagged_tree, newdata = test)
}
```
In the above code we used a bootstrap aggregation bagging techinque to solve the problem of imbalanced data.

To see the accuracy of the model we can plot the number of trees used in the bagging with the RMSE as the following
```
predictions %>%
  as.data.frame() %>%
  mutate(
    observation = 1:n(),
    actual = test$Risk1Yr) %>%
  tidyr::gather(tree, predicted, -c(observation, actual)) %>%
  group_by(observation) %>%
  mutate(tree = stringr::str_extract(tree, '\\d+') %>% as.numeric()) %>%
  ungroup() %>%
  arrange(observation, tree) %>%
  group_by(observation) %>%
  mutate(avg_prediction = cummean(predicted)) %>%
  group_by(tree) %>%
  summarize(RMSE = RMSE(avg_prediction, actual)) %>%
  ggplot(aes(tree, RMSE)) +
  geom_line() +
  xlab('Number of trees')

stopCluster(cl)
```




