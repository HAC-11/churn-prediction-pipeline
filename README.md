[README.md](https://github.com/user-attachments/files/26905103/README.md)
# Subscription Churn Prediction Pipeline

End-to-end machine learning pipeline for predicting customer churn on subscription data. Built as a code sample demonstrating feature engineering, imbalanced data handling, model training, evaluation, and experiment tracking.

---

## Overview

This project builds a binary classifier to predict whether a subscription user will churn. The pipeline covers every stage from raw data to a tracked, evaluated model — including handling the class imbalance problem common in real churn datasets.

**Models:** XGBoost, LightGBM  
**Best CV AUC:** 0.8644 (LightGBM)  
**Dataset:** Synthetic subscription data (5,000 records, ~22% churn rate)

---

## Pipeline Steps

1. **Synthetic data generation** — 5,000 realistic subscription records with engineered churn probability
2. **Exploratory data analysis** — feature distributions split by churn label
3. **Feature engineering** — 15 features across recency, engagement, friction, plan tier, and interaction groups
4. **Train/test split + SMOTE** — stratified split with minority-class oversampling on training data only
5. **Model training** — XGBoost and LightGBM with tuned hyperparameters
6. **Cross-validation** — 5-fold stratified CV for stable performance estimates
7. **Evaluation** — ROC curve, Precision-Recall curve, classification report
8. **Feature importance** — gain-based importance comparison across both models
9. **MLflow tracking** — hyperparameters, metrics, and model artifacts logged per run

---

## Feature Engineering

| Group | Features |
|---|---|
| Recency | `days_since_active`, `log_days_since_active` |
| Engagement | `sessions_per_day`, `log_total_sessions`, `engagement_tier` |
| Friction | `payment_failure_rate`, `support_ticket_rate`, `has_payment_failure`, `has_support_ticket`, `high_friction` |
| Plan | `plan_encoded` |
| Interaction | `recency_x_low_engagement`, `tenure_weighted_sessions` |

---

## Results

| Model | ROC-AUC | CV AUC (mean ± std) |
|---|---|---|
| XGBoost | 0.6399 | 0.8593 ± 0.0020 |
| LightGBM | 0.6397 | 0.8644 ± 0.0035 |

---

## Tech Stack

- Python 3.10
- XGBoost, LightGBM
- Scikit-learn, imbalanced-learn
- Pandas, NumPy, Matplotlib
- MLflow

---

## How to Run

**Install dependencies:**
```bash
pip install xgboost lightgbm imbalanced-learn mlflow scikit-learn pandas matplotlib
```

**Run the notebook:**
```bash
jupyter notebook churn_prediction_pipeline.ipynb
```
Then: **Kernel → Restart & Run All**

**View MLflow experiments:**
```bash
mlflow ui
```
Open `http://localhost:5000` in your browser.

---

## Author

Hetvi Chavda  
MS Data Analytics Engineering — Northeastern University  
chavda.h@northeastern.edu
