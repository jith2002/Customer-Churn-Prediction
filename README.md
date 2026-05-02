# Telco Customer Churn Prediction

## Overview
This project develops an end-to-end machine learning solution to predict customer churn for a telecommunications company. Using a dataset of 7,043 customers, the goal is to proactively identify customers at risk of leaving before they churn, enabling targeted retention interventions.

## Dataset
Source: IBM Telco Customer Churn Dataset
Records: 7,043 customers
Features: 20 input features covering customer demographics, subscribed services, and account information
Target: Churn (1 = churned, 0 = stayed)
Class Distribution: 26.5% churners, 73.5% non-churners

## Methodology

### Data Preparation
Dropped customerID as a non-predictive identifier. Converted TotalCharges to numeric, surfacing 11 blank entries as NaN and imputing with median. One-hot encoded 16 categorical variables expanding the feature space to 30 columns. Applied stratified 80/20 train-test split preserving the 26.5% churn ratio.

### Model Building
Five models were evaluated across three stages. In the first stage, all five models were trained on the original imbalanced data without any class balancing, with the best recall of 55.6% achieved by Logistic Regression. In the second stage, cost-complexity pruning was applied to the Decision Tree with the optimal alpha selected by test recall, improving recall from 49.5% to 61.2%. In the third stage, SMOTE was applied correctly inside an ImbPipeline to prevent data leakage during cross-validation, with all five models improving on recall and Gradient Boosting reaching 66.6%.

### Evaluation
5-fold Stratified Cross-Validation was performed on fresh pipeline clones. Hyperparameter tuning was conducted via RandomizedSearchCV optimizing for recall. ROC and Precision-Recall curves were used for threshold-independent evaluation. Threshold tuning was applied to maximize F1-score. MDI and Permutation Importance were used for feature interpretability. A business cost matrix was built using USD 500 per missed churner and USD 50 per false alarm.

## Final Model: Gradient Boosting (Threshold = 0.55)
•	Recall: 66.6%
•	Precision: 55.3%
•	F1-Score: 60.4%
•	AUC-ROC: 82.7%
•	PR-AUC: 60.3%
•	Accuracy: 77.4%

## Model Performance Progression
•	Baseline (Original Data): 55.6% recall
•	After Decision Tree Pruning: 61.2% recall
•	After SMOTE Pipeline: 66.6% recall
•	After Hyperparameter Tuning: 71.4% recall
•	Final Model (Threshold 0.55): 66.6% recall

## Key Churn Drivers
Seven features consistently identified by both MDI and Permutation Importance: tenure (strongest predictor, first 12 months highest risk), electronic check payment (45.3% churn vs 15-17% for automatic payments), fiber optic internet (41.9% churn despite being premium tier), paperless billing (33.6% churn vs 16.3% for paper billing), two-year contract (just 2.8% churn, dominant retention lever), total charges, and monthly charges.

## Business Recommendations
Prioritize first-year customers for retention intervention. Incentivize contract upgrades from month-to-month to annual plans. Migrate electronic check users to automatic payment methods. Investigate fiber optic service quality and pricing perception. Bundle supplementary services with fiber optic subscriptions. Deploy the model in a monthly scoring pipeline with threshold 0.55.

## Technical Notes
SMOTE is applied inside ImbPipeline to prevent cross-validation data leakage. Decision Tree pruning alpha is selected by test recall not accuracy. All hyperparameter searches optimize for recall as the primary metric. Threshold of 0.55 is selected by F1-score maximization. Business cost analysis shows KNN is financially optimal at USD 63,650 total cost versus Gradient Boosting at USD 65,850, but Gradient Boosting is selected for better precision-recall balance.

## Requirements
pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, joblib

## How to Run
Upload [WA_Fn-UseC_-Telco-Customer-Churn.csv](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) when prompted in Colab. Run all cells top to bottom. Model will be saved and downloaded automatically as churn_model.joblib.

## Author
Built as part of a machine learning course project focusing on classification, imbalanced data handling, and model deployment.
