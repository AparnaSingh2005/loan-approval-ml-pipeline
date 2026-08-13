## Day 3 — Baseline and Logistic Regression

Two classification approaches were evaluated:

1. **Dummy Classifier**
   - Used as a simple baseline.
   - Predicts the majority class and provides a minimum benchmark.

2. **Logistic Regression**
   - Used as the first machine learning model.
   - Combined with the preprocessing pipeline to ensure preprocessing and training remain reproducible.

### Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The Logistic Regression model was compared against the Dummy Classifier to verify that the machine learning model performs better than a simple majority-class baseline.
