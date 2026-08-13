import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "loan_default.csv"

TARGET = "Loan_Status"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["Loan_ID"])

X = df.drop(columns=[TARGET])

y = df[TARGET].map({
    "Y": 1,
    "N": 0
})


# ==========================================
# FEATURE TYPES
# ==========================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "str"]
).columns.tolist()


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ==========================================
# PREPROCESSING
# ==========================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

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


# ==========================================
# RANDOM FOREST PIPELINE
# ==========================================

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            )
        )
    ]
)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Random Forest...")

random_forest_pipeline.fit(
    X_train,
    y_train
)


# ==========================================
# PREDICTIONS
# ==========================================

y_pred = random_forest_pipeline.predict(X_test)

y_prob = random_forest_pipeline.predict_proba(
    X_test
)[:, 1]


# ==========================================
# METRICS
# ==========================================

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


print("\n" + "=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# SAVE RESULTS
# ==========================================

results = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1],
    "ROC-AUC": [roc_auc]
})

results.to_csv(
    BASE_DIR / "outputs" / "day4_random_forest_results.csv",
    index=False
)

print("\nDay 4 completed successfully!")