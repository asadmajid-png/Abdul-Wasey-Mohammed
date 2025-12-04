Bank Marketing Machine Learning Project
Overview

This repository contains the coursework implementation for the 7072CEM Machine Learning module.
The purpose of this project is to use machine learning techniques to predict whether a customer will subscribe to a term deposit based on historical marketing campaign data.

The work includes supervised and unsupervised learning approaches, model evaluation, and result analysis.

Dataset Information

Source: UCI Machine Learning Repository

Name: Bank Marketing Dataset

Records: 41,188 entries

Type: Binary classification

Target variable: y (customer subscription: yes/no)

The dataset contains demographic details, economic indicators, customer financial information, and previous campaign interactions.

Dataset link:
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

Methods

The following machine learning methods were applied:

Supervised Learning Models

Logistic Regression

K-Nearest Neighbours (KNN)

Support Vector Machine (SVM) with RBF kernel

Unsupervised Learning

K-Means Clustering for customer segmentation

Implementation Workflow

Data loading and exploration

Data preprocessing:

Handling categorical variables using One-Hot Encoding

Feature scaling using StandardScaler

Stratified train-test split (70% training, 30% testing)

Model training and validation

Performance evaluation using statistical metrics

Data visualisation including ROC curves and clustering output using PCA

Results Summary

Model performance was evaluated using accuracy, precision, recall, F1-score, and ROC-AUC score.
A summary of the main model performance (ROC-AUC) is shown below:

Model	ROC-AUC Score
Logistic Regression	~0.94
Support Vector Machine	~0.93
K-Nearest Neighbours	~0.82

K-Means clustering identified four meaningful clusters, demonstrating distinct behavioural patterns among customers.

Files Included
File Name	Description
bank_marketing_ml.py	Python script including preprocessing, modelling, and evaluation
README.md	Project description and documentation
Future Work

Several improvements could enhance the performance and scalability of this work:

Applying hyperparameter optimisation such as GridSearchCV

Addressing class imbalance using SMOTE or alternative sampling techniques

Testing additional machine learning models such as Random Forest, XGBoost, and Gradient Boosting

Implementing model deployment through Flask, Streamlit, or similar frameworks

Creating a dashboard for non-technical users to interact with model predictions

Citation

If using the dataset, please cite the following publication:

Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems.

Author

Abdul Wasey Mohammed
7072CEM Machine Learning
Coventry University
