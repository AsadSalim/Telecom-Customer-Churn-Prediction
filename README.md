# Telecom-Customer-Churn-Prediction

## Project Overview
* This is an end-to-end project to predict if customers will churn for a Telecom Company to help companies proactively reach out to unhappy customers and provide better offers to increase customer retention.
* Telco Customer Churn: is our dataset. Which is really popular in Kaggle with over 880 notebooks
* Engineered features from the text of each job description to quantify the value companies put on python, excel, aws, and spark.
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model.
* Built a client facing API using flask

## Code and Resources Used
* Dataset: Telco Customer Churn https://www.kaggle.com/datasets/blastchar/telco-customer-churn
* Python Version: 3.9
* Packages: pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle
* Cleaning: https://stackoverflow.com/questions/13445241/replacing-blank-values-white-space-with-nan-in-pandas
* For Web Framework Requirements: pip install -r requirements.txt
* Flask Productionization: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Data Cleaning
In this section we loaded the data from the .csv file into a Pandas DataFrame. We first changed the <code>TotalCharges</code> column from <code>string</code> to <code>float</code>. Then we dropped all null values and also the <code>customerID</code> column.

## EDA
EDA summary
## Model Building
First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.

I tried three different models:

* Multiple Linear Regression – Baseline for the model
* Lasso Regression – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
* Random Forest – Again, with the sparsity associated with the data, I thought that this would be a good fit.

## Model performance

The Random Forest model far outperformed the other approaches on the test and validation sets.

* Random Forest : MAE = 
* Linear Regression: MAE = 
* Ridge Regression: MAE = 

## Model Deployment
