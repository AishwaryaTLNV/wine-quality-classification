a. Problem Statement

The objective of this assignment is to build and compare multiple machine learning classification models to predict the quality of wine based on its physicochemical properties.
Wine quality prediction is a multi-class classification problem, where the target variable represents discrete quality scores.
The assignment also demonstrates a complete end-to-end ML workflow, including:
Data preprocessing
Model training and evaluation
Building an interactive Streamlit web application
Deploying the application on Streamlit Community Cloud

b. Dataset Description

Dataset Name: Wine Quality Dataset
Source: UCI Machine Learning Repository
Type: Multi-class classification

Dataset Details:
    Total Instances: 6,497 (Red + White wine combined)
    Total Features: 12
    Target Variable: quality (values range from 3 to 9)

Feature Description:
The dataset contains physicochemical attributes of wine such as:
    Fixed acidity
    Volatile acidity
    Citric acid
    Residual sugar
    Chlorides
    Free sulfur dioxide
    Total sulfur dioxide
    Density
    pH
    Sulphates
    Alcohol
    Wine type (red/white encoded as binary)

c. Models Used and Performance Comparison

The following six classification models were implemented and evaluated on the same dataset:
    Logistic Regression
    Decision Tree Classifier
    K-Nearest Neighbors (kNN)
    Naive Bayes (Gaussian)
    Random Forest (Ensemble)
    XGBoost (Ensemble)

Evaluation Metrics Used:
    Accuracy
    AUC Score (One-vs-Rest, weighted)
    Precision (weighted)
    Recall (weighted)
    F1 Score (weighted)
    Matthews Correlation Coefficient (MCC)

Model Comparison Table

ML Model	                Accuracy	AUC	   Precision  Recall  F1 Score	 MCC

Logistic Regression	           0.541	0.723	0.542	   0.541	0.513	0.271
Decision Tree	               0.595	0.700	0.594	   0.595	0.594	0.399
kNN	                           0.558	0.738	0.544	   0.558	0.549	0.326
Naive Bayes	                   0.322	0.597	0.421	   0.322	0.362	0.100
Random Forest (Ensemble)	   0.692	0.866	0.691	   0.692	0.679	0.525
XGBoost (Ensemble)	           0.653	0.837	0.646	   0.653	0.645	0.471

Observations on Model Performance

ML Model	              Observation
Logistic Regression	      Performs moderately due to linear decision boundaries, which are insufficient for complex patterns in the data
Decision Tree	          Captures non-linear relationships but is prone to overfitting
kNN	                      Sensitive to feature scaling and class imbalance
Naive Bayes	              Performs poorly due to strong independence assumptions among features
Random Forest (Ensemble)  Achieves the best overall performance with strong generalization and high MCC
XGBoost (Ensemble)	      Performs competitively with Random Forest and effectively models complex feature interactions

Overall Observation:
Ensemble models outperform individual classifiers, demonstrating better generalization and robustness for multi-class wine quality prediction.
