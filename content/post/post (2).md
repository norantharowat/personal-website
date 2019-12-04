+++
title = "Machine learning project"
date = "2018-12-11"
tags = ["random1", "random2"]
categories = ["Category 11"]
description = "A brief summary for the post 2"
+++

# Introduction

It is time to speak about our first Machine Learning project. Me and my friends are team mates at post-operative life expectancy of lung cancer project, we try to work on lung cancer as it is considered as one of the most common forms of cancer in today’s world. Thoracic surgery is one of the ways to diagnose lung cancer if it is detected at an early stage. Hence it is better to cure lung cancer at the beginning stage. Patients survival cannot be predicted by the surgery alone. Hence if the patient’s survival cannot be extended for a year after surgery, then the factors for the death remains a mystery so in order to overcome this problem we are here used data mining techniques in our project to detect the patient’s survival. This will help in taking the decision to enter the surgery or not. 

# Data and Methodology

In our project we used different Methods for Machine Learning. To detect post-operative life expectancy of lung firstly, we collect data that represents specific features related to the patient case like Forced vital capacity, Cough before surgery, Weakness before surgery, Hemoptysis before surgery and so on using

 http://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data

to get our data then we encoding our data to be easy for using in our project, I have to mention that our data was unbalanced data so this was a challenge for us to deal with it.

We start with preprocessing stage so in this stage we import our data, catch the outliers and remove it because it was obviously while charting the data, renaming the features to be easier for use then data scaling and finally splitting it to test set and train set. For the processing stage , we tried two methods the first one linear regression because it is mainly deal with predictive analysis but unfortunately our model was not accurate enough because as I mentioned above our data was unbalance so the model trained more on the false data than true one for this reason we decided to use Bootstrapping aggregation ( Bagging ) to solve this problem
hence Bootstrapping aggregation divide the data to group of clusters then train the model on each cluster and finally aggregate all clusters together at the end, we have to mention that Bagging helps us too much to avoid the problem of unbalanced data.

![alt text](content/post/RMSE.PNG)

For the second method we used the decision tree method to find out if its result will be better than linear regression method or not, but we faced the same problem because of unbalanced data so we finally decided to use linear regression with Bagging to solve this problem.

# Evaluation
Actually, we used two ways to evaluate our Model
The first one is Confusion Matrix with linear regression to estimate the accuracy, sensitivity, precision and prevalence, before using bagging we had a problem with FN term it was so high and this was a big problem for our model but we solved this problem using Bagging.
The second way is Root Mean Square (RMS) with decision tree method and we got a reasonable error.