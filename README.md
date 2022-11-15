# Telecom-Customer-Churn-Prediction

## Project Overview
* This is an end-to-end project to predict if customers will churn for a Telecom Company to help companies proactively reach out to unhappy customers and provide better offers to increase customer retention.
* Telco Customer Churn: is our dataset. Which is really popular in Kaggle with over 880 notebooks
* Data was cleaned. Then we carried out extensive EDA on it using Numpy, Pandas, Matplotlib, Seaborn and Dython. 
* Optimized XGBoost model using RandomizedSearchCV to reach the best model.
* Built a client facing API using flask

## Code and Resources Used
* Dataset: Telco Customer Churn https://www.kaggle.com/datasets/blastchar/telco-customer-churn
* Python Version: 3.9
* Packages:* Packages: 
    - Pandas:provides data analysis/manipulation tools
    - Numpy: fundamental package for scientific computing in Python.
    - Matplotlib: Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
    - Seaborn: based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
    - Dython: provides a set of data analysis tools for Python 3.X. 
    - Sklearn: a library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data             preprocessing, model selection, model evaluation, and many other utilities.
    - imblearn: a toolbox to deal handle imbalanced datasets in machine learning.
    - xgboost:  is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
    - pickle: The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
    - flask:  a Python web framework built with a small core and easy-to-extend philosophy.
* Cleaning: https://stackoverflow.com/questions/13445241/replacing-blank-values-white-space-with-nan-in-pandas
* Investigating association among categorical features: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
* Model selection automation: https://www.youtube.com/watch?v=7uLzGRlXXDw
* For Web Framework Requirements: pip install -r requirements.txt
* Flask Productionization: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Data Cleaning
Since the dataset is from Kaggle, this saved us a lot of work and it did not need much effort to clean. In this section we loaded the data from the .csv file into a Pandas DataFrame then carried out the following steps:
    - The 'TotalCharges' column is converted from an object type to float64.
    - Spaces as ' ' is replaced by Null object which Pandas can recognize.
    - Null values was dropped as they had a small percentage with regard to the Dataframe dimensions.
    - Drop the customerID column since it's irrelevant.

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
