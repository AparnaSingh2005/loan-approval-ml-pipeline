## Day 6 — Model Diagnostics and Failure Analysis

The final model was evaluated on the held-out test set.

### Diagnostics

The evaluation included:

- Classification report
- Confusion matrix
- ROC curve and ROC-AUC score
- Feature importance analysis
- Incorrect prediction analysis

## Failure Analysis

The final model made 11 incorrect predictions out of 77 test samples, resulting in a failure rate of 14.29%.

The confusion matrix showed:

- 14 correctly predicted rejected applications
- 52 correctly predicted approved applications
- 8 rejected applications incorrectly predicted as approved
- 3 approved applications incorrectly predicted as rejected

The model showed stronger performance for the approved class, with a recall of 95%, compared with 64% recall for rejected applications.

Some incorrect predictions also had relatively high predicted approval probabilities, indicating that model confidence does not guarantee correctness.

### Feature Importance

The most influential feature was `Credit_History`, with an importance score of approximately 0.41. Other important features included:

- Loan Amount
- Co-applicant Income
- Applicant Income

The strong dependence on credit history means that missing, inaccurate, or biased credit history information could significantly affect predictions.

## Model Evaluation

The final tuned model was evaluated on a held-out test set of 77 samples.

| Metric | Score |
|---|---:|
| Accuracy | 0.86 |
| ROC-AUC | 0.8149 |
| Rejected F1-score | 0.72 |
| Approved F1-score | 0.90 |
| Incorrect Predictions | 11 |
| Failure Rate | 14.29% |

### Classification Performance

The model performed better at predicting approved loans than rejected loans.

- Approved recall: **95%**
- Rejected recall: **64%**

This difference is likely influenced by the class imbalance in the dataset, where approximately 71% of applications were approved and 29% were rejected.

Therefore, accuracy alone was not used to evaluate the model. Precision, recall, F1-score, and ROC-AUC were also considered.


## Model Limitations

This model has several important limitations:

1. The dataset contains only 381 records, so the evaluation results may vary with different train-test splits.

2. The dataset has a class imbalance, with approximately 71% approved and 29% rejected applications.

3. The model performs better at identifying approved applications than rejected applications.

4. The model relies heavily on `Credit_History`, so missing or inaccurate values for this feature may significantly affect predictions.

5. Important real-world variables such as detailed credit scores, existing debt, employment stability, repayment history, and economic conditions are not included.

6. Historical approval data may contain bias that can be learned and reproduced by the model.

7. The model should not be used as the sole basis for real-world lending decisions.
