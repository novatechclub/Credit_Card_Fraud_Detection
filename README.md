# Logistic Regression – Fraud Detection

## Overview
This project applies a logistic regression model to detect fraudulent transactions using a transaction dataset.

## Baseline Model
Initial model performance:

- Accuracy: 99.05%
- Fraud Recall: 45%
- Fraud Precision: 88%

Although the baseline model achieved very high accuracy, it missed more than half of the fraudulent transactions because the dataset is imbalanced.

## Improved Model
To address class imbalance, the logistic regression model was updated using class weighting:

```python
LogisticRegression(max_iter=1000, class_weight="balanced")

import pandas as pd


