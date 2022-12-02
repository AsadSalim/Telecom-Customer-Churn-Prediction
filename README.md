# Telecom-Customer-Churn-Prediction

## Project Overview

- This is an end-to-end project to predict if customers will churn for a Telecom Company to help companies proactively reach out to unhappy customers and provide better offers to increase customer retention.
- Telco Customer Churn: is our dataset. Which is really popular in Kaggle with over 880 notebooks
- Engineered features from the text of each job description to quantify the value companies put on python, excel, aws, and spark.
- Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model.
- Built a client facing API using flask

## Code and Resources Used

- Dataset: Telco Customer Churn https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Python Version: 3.9
- Packages: pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle
- Cleaning: https://stackoverflow.com/questions/13445241/replacing-blank-values-white-space-with-nan-in-pandas
- Investigating association among categorical features: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
- For Web Framework Requirements: pip install -r requirements.txt
- Flask Productionization: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Data Cleaning

In this section we loaded the data from the .csv file into a Pandas DataFrame. We first changed the <code>TotalCharges</code> column from <code>string</code> to <code>float</code>. Then we dropped all null values and also the <code>customerID</code> column.

## EDA

EDA summary

## Model Building

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.

I tried three different models:

- Multiple Linear Regression – Baseline for the model
- Lasso Regression – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
- Random Forest – Again, with the sparsity associated with the data, I thought that this would be a good fit.

## Model performance

The Random Forest model far outperformed the other approaches on the test and validation sets.

- Random Forest : MAE =
- Linear Regression: MAE =
- Ridge Regression: MAE =

## Model Deployment

After modeling the data and export the model using pickle, we have reach to our final step of this project - deployment. In order to make our model available online for everyone to try and use, we need several tools and concepts from the web development field.
We used Flask as a backend to serve the API endpoints and basic HTML and tailwindcss to render the frontend UI. And to make the website available on the internet we used render.com to host it.

First, we created a Flask app, then created 3 API endpoints. The "/" endpoint is the home page, which will render the landing page with the project’s info along with our contact details. And the "/predict" endpoint will redirect you to the input form, which you can submit all the features you want to predict the churn of. The last "/result" endpoint will render the same prediction page but with the result of the prediction in the bottom.

TO deploy the website, we used render.com, which connects directly to our GitHub repo. It has easy steps to deploy your projects for free. We originally wanted to use Heroku, but sadly it terminated it's free tier options.

All and all, this step was very fun and informative. It gave us a glimpse of what a Machine Learning Engineer job is, and how important to communicate with the rest of the team to make sure the project see the light.
