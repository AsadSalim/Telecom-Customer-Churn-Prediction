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
* The 'TotalCharges' column is converted from an object type to float64.
* Spaces as ' ' is replaced by Null object which Pandas can recognize.
* Null values was dropped as they had a small percentage with regard to the Dataframe dimensions.
* Drop the customerID column since it's irrelevant.

## EDA
Used Matplotlib and Seaborn graphs to explore some of the important columns to gain some understanding of them. In addition to that Dython was used to
carry out analysis using the Categorical Features i.e. explore association among categorical features. Finally we used Pandas to answer some questions
related to the dataset.

## Model Building
Before training the model, we carried out some cleaning to the data to ensure that the model can fully understand it. Then we used One Hot Encoder to  transform categorical variables into dummy variables. We also split the data into input and output. Then we handled the unbalance in the data using 
down sampling to ensure better performance without leaning to one category over the other.

Then we started testing Three models: Decision Trees, Random Forests and XGBoost. It turned out that XGBoost is superior to the other two. So we took it further to perform hyperparameter tuning to get optimized performance. This step was done using RandomizedSearchCV from the Sklearn's model_selection module.

Before saving the model it was critical to train the model with the best obtained parameters using the whole data. Otherwise we would be throughing away potentially valuable data that the model can learn from.

Finally we saved the trained model using the Pickle method for later use without the need to retrain it again and again. This file was saved on the same working directory and will be used again in the deployment phase.

## Model performance

The XG Boost outperformed the other approaches on the test and validation sets as follows:

### Decision Tree:
- Accuracy: 0.9304
- F1 score: 0.9303
- Precision: 0.9215
- Recall: 0.9457
- ROC AUC score: 0.9299

### Random Forest:
- Accuracy: 0.9421
- F1 score: 0.9421
- Precision: 0.944
- Recall: 0.944
- ROC AUC score: 0.9421

### XG Boost:
- Accuracy: 0.9512
- F1 score: 0.9512
- Precision: 0.9465
- Recall: 0.9597
- ROC AUC score: 0.9509

## Model Deployment
After modeling the data and export the model using pickle, we have reach to our final step of this project - deployment. In order to make our model available online for everyone to try and use, we need several tools and concepts from the web development field. We used Flask as a backend to serve the API endpoints and basic HTML and tailwindcss to render the frontend UI. And to make the website available on the internet we used render.com to host it.

First, we created a Flask app, then created 3 API endpoints. The "/" endpoint is the home page, which will render the landing page with the project’s info along with our contact details. And the "/predict" endpoint will redirect you to the input form, which you can submit all the features you want to predict the churn of. The last "/result" endpoint will render the same prediction page but with the result of the prediction in the bottom.

To deploy the website, we used render.com, which connects directly to our GitHub repo. It has easy steps to deploy your projects for free. We originally wanted to use Heroku, but sadly it terminated it's free tier options.

All and all, this step was very fun and informative. It gave us a glimpse of what a Machine Learning Engineer job is, and how important to communicate with the rest of the team to make sure the project see the light.
