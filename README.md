# Customer Churn Prediction in the Telecom Industry
## Author: Pragya Mishra
# Overview
Predicting customer churn is essential for the telecom sector's ability to retain customers and maintain revenue. Given the fierce competition and diverse customer base, it's critical to identify the reasons for customer churn and address them proactively. This project involves predicting customer churn using various machine learning models, including Decision Tree, Random Forest, SVM, AdaBoost, and Gradient Boosting classifiers.

# Project Structure
The project is structured as follows:

# Data Loading and Preprocessing: Loading the dataset and performing necessary preprocessing steps.
# Exploratory Data Analysis (EDA): Understanding the data distribution and identifying important features.
# Feature Engineering: Transforming categorical variables and standardizing numerical features.
# Model Building and Evaluation: Building and evaluating multiple machine learning models.
# Conclusion: Summarizing the results and identifying the best-performing model.

#Dependencies
pandas
numpy
missingno
matplotlib
seaborn
plotly
scikit-learn
xgboost
catboost

# Data Preprocessing
Handling Missing Values: Visualized and managed missing values in the dataset.
Feature Encoding: Applied label encoding and one-hot encoding to categorical variables.
Feature Scaling: Standardized numerical features for better model performance.
Exploratory Data Analysis (EDA)
Gender and Churn Distribution: Analyzed the distribution of gender and churn using pie charts.
Payment Method Analysis: Visualized the distribution of payment methods and their relationship with churn.
Monthly Charges and Churn: Examined the relationship between monthly charges and churn using KDE plots and box plots.

# Machine Learning Models
K-Nearest Neighbors (KNN):

Optimal value of k: 30
Accuracy: 74.2%
ROC-AUC: 0.83
Support Vector Machine (SVM):

Accuracy: 79.4%
ROC-AUC: 0.84
Decision Tree Classifier:

Accuracy: 74.5%
AdaBoost Classifier:

Accuracy: 80.1%
Gradient Boosting Classifier:

Accuracy: 80.5%
Conclusion
From the modeling and evaluation, we conclude that the best model for predicting customer churn in the telecom industry is the Gradient Boosting Classifier, with an accuracy of approximately 80.5%.

# Results
The project includes the ROC curves, confusion matrices, and classification reports for all the models evaluated. Detailed visualizations and analysis are provided to understand the performance of each model.

# Future Work
Feature Selection: Perform feature selection to further improve model performance.
Hyperparameter Tuning: Experiment with more hyperparameters to optimize the models.
Ensemble Methods: Explore other ensemble methods like stacking to combine multiple models for better performance.
# Contact
For any questions or feedback, feel free to contact me at pragyamis33@gmail.com.
