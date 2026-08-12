import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")


# ==================================================
# DROP IDENTIFIER COLUMN
# ==================================================

print("\n" + "=" * 60)
print("DROPPING IDENTIFIER COLUMN")
print("=" * 60)

df = df.drop(columns=["Loan_ID"])

print("Dropped: Loan_ID")


# ==================================================
# SEPARATE FEATURES AND TARGET
# ==================================================

print("\n" + "=" * 60)
print("SEPARATING FEATURES AND TARGET")
print("=" * 60)

X = df.drop(columns=[TARGET])

# Convert target:
# Y -> 1 (Approved)
# N -> 0 (Rejected)

y = df[TARGET].map({
    "Y": 1,
    "N": 0
})

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget distribution (%):")
print((y.value_counts(normalize=True) * 100).round(2))


# ==================================================
# IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ==================================================

print("\n" + "=" * 60)
print("IDENTIFYING FEATURE TYPES")
print("=" * 60)

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "str"]
).columns.tolist()


print("\nNumerical Features:")
for column in numerical_features:
    print(f"- {column}")

print("\nCategorical Features:")
for column in categorical_features:
    print(f"- {column}")


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

print(f"\nTraining target shape: {y_train.shape}")
print(f"Testing target shape: {y_test.shape}")


print("\nTraining target distribution:")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTesting target distribution:")
print((y_test.value_counts(normalize=True) * 100).round(2))


# ==================================================
# NUMERICAL PREPROCESSING PIPELINE
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
# CATEGORICAL PREPROCESSING PIPELINE
# ==================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ==================================================
# COMBINE PREPROCESSING
# ==================================================

print("\n" + "=" * 60)
print("BUILDING PREPROCESSING PIPELINE")
print("=" * 60)

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

print("Preprocessing pipeline created successfully!")


# ==================================================
# FIT ON TRAINING DATA ONLY
# ==================================================

print("\n" + "=" * 60)
print("PREPROCESSING DATA")
print("=" * 60)

# Learn preprocessing parameters ONLY from training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformations to test data
X_test_processed = preprocessor.transform(X_test)


print("Training data processed successfully!")
print("Testing data processed successfully!")

print(f"\nProcessed training shape: {X_train_processed.shape}")
print(f"Processed testing shape: {X_test_processed.shape}")

# ==================================================
# VERIFICATION CHECKS
# ==================================================

print("\n" + "=" * 60)
print("VERIFICATION CHECKS")
print("=" * 60)

print(f"Missing values in target: {y.isnull().sum()}")

# Check missing values after preprocessing
X_train_dense = X_train_processed
X_test_dense = X_test_processed

print(
    "Missing values in processed training data:",
    pd.DataFrame(X_train_dense).isnull().sum().sum()
)

print(
    "Missing values in processed testing data:",
    pd.DataFrame(X_test_dense).isnull().sum().sum()
)

# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("DAY 2 PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Original dataset shape: {df.shape}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Original features: {X.shape[1]}")
print(f"Features after preprocessing: {X_train_processed.shape[1]}")

print("\nNo preprocessing information was learned from the test set.")
print("This helps prevent data leakage.")