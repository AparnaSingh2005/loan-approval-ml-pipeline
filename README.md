## Day 6 — Model Diagnostics and Failure Analysis

The final model was evaluated on the held-out test set.

### Diagnostics

The evaluation included:

- Classification report
- Confusion matrix
- ROC curve and ROC-AUC score
- Feature importance analysis
- Incorrect prediction analysis

### Failure Analysis

All incorrect predictions were extracted and saved for manual inspection. The analysis focused on identifying potential patterns among false positives and false negatives.

Because the dataset is relatively small and moderately imbalanced, model performance was not evaluated using accuracy alone. Precision, recall, F1-score, and ROC-AUC were also considered.