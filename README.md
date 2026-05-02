# Telco Customer Churn Prediction

An end-to-end machine learning solution to predict customer churn for a telecommunications company. Using a dataset of 7,043 customers, the goal is to proactively identify customers at risk of leaving before they churn, enabling targeted retention interventions.

---

## Dataset

| Property | Detail |
|---|---|
| Source | IBM Telco Customer Churn Dataset |
| Records | 7,043 customers |
| Features | 20 input features (demographics, services, account info) |
| Target | Churn (1 = churned, 0 = stayed) |
| Class Distribution | 26.5% churners, 73.5% non-churners |

---

## Methodology

### Data Preparation
- Dropped `customerID` as a non-predictive identifier
- Converted `TotalCharges` to numeric, surfacing 11 blank entries as NaN and imputing with median
- One-hot encoded 16 categorical variables, expanding the feature space to 30 columns
- Applied stratified 80/20 train-test split preserving the 26.5% churn ratio

### Model Building

Five classifiers were evaluated across three progressive stages:

| Stage | Approach | Best Recall |
|---|---|---|
| 1 | Baseline — original imbalanced data, no corrections | 55.6% (Logistic Regression) |
| 2 | Cost-complexity pruning on Decision Tree, alpha selected by test recall | 61.2% |
| 3 | SMOTE inside `ImbPipeline` to prevent cross-validation leakage | 66.6% (Gradient Boosting) |

### Evaluation
- 5-fold Stratified Cross-Validation on fresh pipeline clones
- Hyperparameter tuning via `RandomizedSearchCV` optimizing for recall
- ROC and Precision-Recall curves for threshold-independent evaluation
- Threshold tuning to maximize F1-score
- MDI and Permutation Importance for feature interpretability
- Business cost matrix: USD 500 per missed churner, USD 50 per false alarm

---

## Final Model: Gradient Boosting (Threshold = 0.55)

| Metric | Score |
|---|---|
| Recall | 66.6% |
| Precision | 55.3% |
| F1-Score | 60.4% |
| AUC-ROC | 82.7% |
| PR-AUC | 60.3% |
| Accuracy | 77.4% |

### Performance Progression

| Stage | Recall |
|---|---|
| Baseline (original data) | 55.6% |
| After Decision Tree pruning | 61.2% |
| After SMOTE pipeline | 66.6% |
| After hyperparameter tuning | 71.4% |
| Final model (threshold 0.55) | 66.6% |

---

## Key Churn Drivers

Seven features were consistently identified by both MDI and Permutation Importance:

1. **Tenure** — strongest predictor; first 12 months represent highest churn risk
2. **Electronic check payment** — 45.3% churn rate vs. 15–17% for automatic payments
3. **Fiber optic internet** — 41.9% churn rate despite being the premium service tier
4. **Paperless billing** — 33.6% churn vs. 16.3% for paper billing customers
5. **Two-year contract** — just 2.8% churn; dominant retention lever in the dataset
6. **Monthly charges** — higher billing amounts correlate with increased churn risk
7. **Total charges** — driven largely by tenure but carries independent signal

---

## Business Recommendations

1. **Prioritize first-year customers** for proactive retention intervention; median churner tenure is just 10 months
2. **Incentivize contract upgrades** from month-to-month to annual plans; churn drops from 42.7% to 2.8%
3. **Migrate electronic check users** to automatic payment methods via billing credits or incentives
4. **Investigate fiber optic service quality** — premium pricing with the highest churn rate signals unmet expectations
5. **Bundle supplementary services** with new fiber optic subscriptions to increase perceived value and stickiness
6. **Deploy in a monthly scoring pipeline** with threshold 0.55; review threshold quarterly based on campaign outcomes

---

## Technical Notes

- SMOTE is applied inside `ImbPipeline` to prevent cross-validation data leakage
- Decision Tree pruning alpha is selected by test recall, not accuracy
- All hyperparameter searches optimize for recall as the primary metric
- Threshold of 0.55 is selected by F1-score maximization over the probability range
- Business cost analysis shows KNN is financially optimal at USD 63,650 total cost vs. Gradient Boosting at USD 65,850 — Gradient Boosting is selected for its stronger precision-recall balance and higher AUC scores

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
```

---

## How to Run

1. Open the notebook in Google Colab
2. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` when prompted
3. Run all cells top to bottom
4. The trained model will be saved and downloaded automatically as `churn_model.joblib`

---

## Author

**Ranjith Kumar T M**  
MBA — Business Analytics & Marketing, PES University  
[LinkedIn](https://linkedin.com/in/ranjith-kumar-t-m) · [GitHub](https://github.com/jith2002)
