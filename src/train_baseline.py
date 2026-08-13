import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==================================================
# CONFIGURATION
# ==================================================

DATA_PATH = "data/loan_default.csv"
TARGET = "Loan_Status"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ==================================================
# LOAD DATA
# ==================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ==================================================
# DROP IDENTIFIER
# ==================================================

df = df.drop(columns=["Loan_ID"])


# ==================================================
# FEATURES AND TARGET
# ==================================================

X = df.drop(columns=[TARGET])

y = df[TARGET].map({
    "Y": 1,
    "N": 0
})


# ==================================================
# IDENTIFY FEATURE TYPES
# ==================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "str"]
).columns.tolist()


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==================================================
# NUMERICAL PREPROCESSING
# ==================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ==================================================
# CATEGORICAL PREPROCESSING
# ==================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# ==================================================
# PREPROCESSOR
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ==================================================
# EVALUATION FUNCTION
# ==================================================

def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


# ==================================================
# 1. DUMMY BASELINE MODEL
# ==================================================

dummy_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            DummyClassifier(
                strategy="most_frequent"
            )
        )
    ]
)

print("\nTraining Dummy Classifier...")

dummy_pipeline.fit(X_train, y_train)

dummy_results = evaluate_model(
    dummy_pipeline,
    X_test,
    y_test,
    "Dummy Classifier"
)


# ==================================================
# 2. LOGISTIC REGRESSION MODEL
# ==================================================

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)

print("\nTraining Logistic Regression...")

logistic_pipeline.fit(X_train, y_train)

logistic_results = evaluate_model(
    logistic_pipeline,
    X_test,
    y_test,
    "Logistic Regression"
)


# ==================================================
# MODEL COMPARISON
# ==================================================

results = pd.DataFrame(
    [
        dummy_results,
        logistic_results
    ]
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results.round(4))


# ==================================================
# SAVE RESULTS
# ==================================================

results.to_csv(
    "outputs/day3_model_comparison.csv",
    index=False
)

print("\nResults saved to:")
print("outputs/day3_model_comparison.csv")


print("\n" + "=" * 60)
print("DAY 3 COMPLETED SUCCESSFULLY")
print("=" * 60)