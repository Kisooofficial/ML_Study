# What is this Repository?
This repository summarizes what I studied while taking the machine learning course, which was the first semester of 2022. Main Course is Machine Learning, but it also includes Data Mining too. 

# What includes?
+ Data Mining Homework => About Odds, Linear Regression, Logistic Regression
+ Machine Learning Homework => About Clustering, Decision Tree
It also contains Python and R Code

# Used Data
+ Data Mining for Linear Regression => https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
+ Data Mining for Logistic Regression => https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?datasetId=306&sortBy=voteCount
+ Machine Learning for Clustering => https://archive.ics.uci.edu/ml/datasets/adult
+ Machine Learning for Decision Tree => https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset

# Details of Data Mining Homework
## Odds
+ Talking about Odds and how odds apply in real life
## Linear Regression
+ Before building a model, Data preprocessing and EDA
+ Build Linear Regression and interpret a model
+ After building first model, do feature selection and compare with others.
+ evaluate model with using MAE, MSE, ANOVA, p-value
## Logistic Regression
+ Before building a model, Data preprocessing and EDA
+ Build Logistic Regression and interpret a model
+ After building first model, do feature selection and compare with others.
+ Find proper Cut-off
+ evaluate model with using MAE, MSE, ANOVA, p-value using confusion matrix, AIC, BIC, p-value <br><br>
<strong>The result was</strong><br>
<img src = "https://user-images.githubusercontent.com/84063359/177470861-c8b37556-34fe-4e88-bd3e-df0c43a9203a.png" width = 80% height = 220></img><br>
<strong>when pred_proba was 0.28, it was best on the basis of f1-score.</strong>

# Details of Machine Learning Homework
## Clustering
+ Data preprocessing and EDA
+ Clustering Analysis with KMeans, hierarchical Clustering, DBSCAN
+ make an intuitive comparison(because of unsupervised learning) and using Silhouette score, V-measure
+ Hyperparameter tuning

## Decision Tree
+ Data preprocessing and EDA
+ make an outlier of 3 percent (Professor's homework requirements)
+ Build a model and compare(ex. if prune? or not?)
+ evaulate model with confusion matrix
+ Interpreting models with graphviz
+ Hyperparameter tuning for gridsearch
+ find important feature for feature_importances

## Decision Tree Results
|thresholds|0.3|0.35|0.4|0.45|0.5|0.55|0.6|
|---|---|---|---|---|---|---|---|
|Accuracy|88.7%|89.0%|89.4%|89.4%|89.3%|89.3%|89.1%|
|Precision|72.5%|67.7%|63.5%|63.5%|62.7%|61.6%|56.2%|
|Recall|61.6%|63.7%|66.4%|66.4%|66.3%|66.5%|67.8%|
|f1-score|66.6%|65.7%|65.0%|65.0%|64.4%|64.0%|61.5%|
|AUC score|82.1%|80.3%|78.8%|78.8%|78.4%|78.0%|75.7%|
