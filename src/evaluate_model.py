import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "loan_default.csv"

MODEL_PATH = BASE_DIR / "models" / "best_loan_model.pkl"

OUTPUT_DIR = BASE_DIR / "outputs"

TARGET = "Loan_Status"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# Create outputs folder if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==================================================
# PREPARE FEATURES AND TARGET
# ==================================================

df = df.drop(columns=["Loan_ID"])

X = df.drop(columns=[TARGET])

y = df[TARGET].map({
    "Y": 1,
    "N": 0
})


# ==================================================
# RECREATE SAME TRAIN / TEST SPLIT
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
# LOAD SAVED MODEL
# ==================================================

print("\n" + "=" * 60)
print("LOADING SAVED MODEL")
print("=" * 60)

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ==================================================
# MAKE PREDICTIONS
# ==================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ==================================================
# CLASSIFICATION REPORT
# ==================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=["Rejected", "Approved"]
)

print(report)


# Save report

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w"
) as file:
    file.write(report)


# ==================================================
# CONFUSION MATRIX
# ==================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Rejected", "Approved"]
)

display.plot()

plt.title("Loan Approval Prediction - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png"
)

plt.close()

print("Confusion matrix saved!")


# ==================================================
# ROC CURVE
# ==================================================

print("\n" + "=" * 60)
print("ROC CURVE")
print("=" * 60)

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print(f"ROC-AUC Score: {roc_auc:.4f}")


plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"Model (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Loan Approval Prediction")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png"
)

plt.close()

print("ROC curve saved!")


# ==================================================
# FAILURE CASE ANALYSIS
# ==================================================

print("\n" + "=" * 60)
print("FAILURE CASE ANALYSIS")
print("=" * 60)


failure_cases = X_test.copy()

failure_cases["Actual"] = y_test.values

failure_cases["Predicted"] = y_pred

failure_cases["Approval_Probability"] = y_prob


# Keep only incorrect predictions

failure_cases = failure_cases[
    failure_cases["Actual"]
    !=
    failure_cases["Predicted"]
]


failure_cases.to_csv(
    OUTPUT_DIR / "failure_cases.csv",
    index=False
)


print(
    f"Total incorrect predictions: {len(failure_cases)}"
)

print(
    f"Failure rate: "
    f"{len(failure_cases) / len(X_test) * 100:.2f}%"
)

print("\nSample failure cases:")

print(
    failure_cases.head()
)


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


# Get preprocessor and trained Random Forest

preprocessor = model.named_steps["preprocessor"]

random_forest = model.named_steps["model"]


# Get processed feature names

feature_names = preprocessor.get_feature_names_out()


# Get importance values

importance_values = random_forest.feature_importances_


feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


feature_importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)


print(
    feature_importance.head(10)
)


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("DAY 6 EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Generated files:")

print("- classification_report.txt")
print("- confusion_matrix.png")
print("- roc_curve.png")
print("- failure_cases.csv")
print("- feature_importance.csv")