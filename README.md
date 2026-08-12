# Loan Approval Prediction

An end-to-end Machine Learning project that predicts whether a loan application is likely to be approved based on applicant and financial information.

## Problem Statement

The goal of this project is to predict the loan approval status of an applicant using demographic, financial, and loan-related features.

The target variable is:

- `Y` → Loan Approved
- `N` → Loan Rejected

## Dataset

The dataset contains 381 records and 13 columns.

Features include:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area

## Project Progress

### Day 1 — Data Exploration

Completed:

- Loaded the dataset using pandas
- Explored dataset shape and columns
- Checked data types
- Identified missing values
- Generated statistical summaries
- Analyzed the target variable
- Created a class distribution visualization

### Key Findings

- Dataset size: **381 rows × 13 columns**
- Target variable: `Loan_Status`
- Identifier column: `Loan_ID`
- Missing values were found in:
  - Gender
  - Dependents
  - Self_Employed
  - Loan_Amount_Term
  - Credit_History

These missing values will be handled in the preprocessing pipeline.

## Tech Stack

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn

## Project Structure

```text
Loan_Default_Prediction/
├── data/
├── src/
│   └── eda.py
├── outputs/
│   └── class_distribution.png
├── models/
├── requirements.txt
└── README.md