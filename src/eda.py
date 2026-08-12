import pandas as pd
import matplotlib.pyplot as plt
import os


# -----------------------------
# LOAD DATASET
# -----------------------------

file_path = "data/loan_default.csv"

df = pd.read_csv(file_path)


# -----------------------------
# BASIC INFORMATION
# -----------------------------

print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)

print(df.head())


print("\n" + "=" * 50)
print("DATASET SHAPE")
print("=" * 50)

print(df.shape)


print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)

print(df.columns.tolist())


print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

df.info()


print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())


print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)

print(df.describe())


# -----------------------------
# TARGET ANALYSIS
# -----------------------------

target = "Loan_Status"

print("\n" + "=" * 50)
print("TARGET DISTRIBUTION")
print("=" * 50)

print(df[target].value_counts())


print("\n" + "=" * 50)
print("TARGET DISTRIBUTION (%)")
print("=" * 50)

print(df[target].value_counts(normalize=True) * 100)


# -----------------------------
# CLASS DISTRIBUTION GRAPH
# -----------------------------

os.makedirs("outputs", exist_ok=True)

df[target].value_counts().plot(kind="bar")

plt.title("Loan Default Distribution")
plt.xlabel("Loan Default")
plt.ylabel("Number of Applicants")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("outputs/class_distribution.png")

plt.show()

print("\nGraph saved successfully!")