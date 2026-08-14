# Loan Approval Prediction — End-to-End Machine Learning Pipeline

## 📌 Project Overview

This project builds an end-to-end machine learning pipeline to predict whether a loan application is likely to be approved or rejected.

The project focuses not only on training a model but also on building a reproducible machine learning workflow, including data exploration, preprocessing, baseline comparison, model training, hyperparameter tuning, evaluation, and failure analysis.

---

## 🎯 Problem Statement

Financial institutions receive many loan applications and need to evaluate multiple factors before making approval decisions.

The goal of this project is to predict:

> **Will a loan application be approved or rejected based on the available applicant information?**

This is a **binary classification problem**.

### Target Variable

`Loan_Status`

- `Y` → Loan Approved
- `N` → Loan Rejected

---

## 📊 Dataset

The dataset contains **381 loan applicant records**.

Features include:

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

The identifier column `Loan_ID` was removed because it does not provide useful predictive information.

---

## 🎯 Evaluation Metrics

The dataset contains a class imbalance:

- Approved: approximately **71%**
- Rejected: approximately **29%**

Therefore, accuracy alone was not sufficient.

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

ROC-AUC was particularly useful for evaluating how well the model separates approved and rejected applications.

---

# 🏗️ Project Structure

```text
loan-approval-ml-pipeline/
│
├── data/
│   └── loan_data.csv
│
├── src/
│   ├── eda.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── train_random_forest.py
│   ├── tune_model.py
│   └── evaluate_model.py
│
├── models/
│   └── best_loan_model.pkl
│
├── outputs/
│   ├── day3_model_comparison.csv
│   ├── day4_random_forest_results.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── failure_cases.csv
│   └── feature_importance.csv
│
├── requirements.txt
├── .gitignore
└── README.md
