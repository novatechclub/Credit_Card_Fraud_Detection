# Credit Card Fraud Detection

**NOVA SBE Tech Club** project on credit card fraud detection. The goal is to explore a dataset of 10,000 transactions, identify the most informative risk signals, and compare two classification models on a strongly imbalanced problem (fraud ~1.5% of the records).

## Repository structure

| File | Description |
|------|-------------|
| [credit_card_fraud_10k.csv](credit_card_fraud_10k.csv) | Dataset: 10,000 transactions, 10 columns, binary target `is_fraud`. |
| [EDA.ipynb](EDA.ipynb) | Exploratory data analysis: cleaning, distributions, fraud rate by variable, correlations. |
| [models.py](models.py) | Definition of the two models (Logistic Regression and Random Forest) with fixed hyperparameters. |
| [evaluation.ipynb](evaluation.ipynb) | Training, evaluation, and comparison of the models on fraud-relevant metrics. |

## Dataset

The file `credit_card_fraud_10k.csv` contains 10,000 transactions with the following columns:

- `transaction_id` — identifier (excluded from analysis)
- `amount` — transaction amount
- `transaction_hour` — hour of the day (0-23)
- `merchant_category` — merchant category (`Food`, `Grocery`, `Clothing`, `Electronics`, `Travel`)
- `foreign_transaction` — foreign transaction flag (0/1)
- `location_mismatch` — geographic inconsistency flag (0/1)
- `device_trust_score` — device trust score (25-99)
- `velocity_last_24h` — number of transactions in the last 24h
- `cardholder_age` — cardholder age
- `is_fraud` — **target** (0/1)

The dataset is already clean: no missing values, no duplicates. The positive class (fraud) accounts for about 1.5% of the records.

## Key EDA findings

- Strongly imbalanced target (~1.5% fraud) → accuracy is misleading; we rely on recall, precision, F2, and AUC-PR.
- `amount` is positively skewed; fraudulent transactions have a slightly higher average amount, but with substantial overlap.
- `velocity_last_24h` is higher in fraudulent transactions → bursts of activity are a meaningful signal.
- Both `foreign_transaction` and `location_mismatch` are associated with a markedly higher fraud rate.
- The **combination** of foreign transaction and location mismatch produces the highest fraud rate: an interaction effect worth retaining for modeling.
- Fraud rate also varies across `merchant_category`, indicating that merchant context matters.

## Models

Defined in [models.py](models.py), both with `class_weight="balanced"` to compensate for the imbalance:

- **Logistic Regression** — `max_iter=1000`, applied to standardized features.
- **Random Forest** — `n_estimators=200`, `max_depth=15`, `min_samples_split=10`, `min_samples_leaf=4`.

## Evaluation

The split is 80/20, stratified on `y` to preserve the fraud proportion. Metrics used:

- **Recall** — share of fraud caught (top priority in a fraud setting).
- **Precision** — reliability of the alerts.
- **F2** — combination that weighs recall twice as much as precision.
- **AUC-PR** — the most informative threshold-independent metric under strong class imbalance.

### Test set results

| Model | Precision | Recall | F2 | ROC-AUC | AUC-PR |
|---------|-----------|--------|-----|---------|--------|
| Logistic Regression | 0.266 | 1.000 | 0.644 | — | 0.738 |
| Random Forest | 1.000 | 0.800 | 0.833 | 1.000 | 0.998 |

- **Logistic Regression** catches every fraud (recall = 1) but generates a high volume of false positives.
- **Random Forest** is far more selective: zero false positives, AUC-PR ~ 1, higher F2. It is the overall stronger model.
- A natural next step is to lower the Random Forest decision threshold below 0.5 to recover part of the 20% of fraud it currently misses, without losing too much precision.

## How to reproduce the results

Requirements:

```
python >= 3.10
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Suggested workflow:

1. Open and run [EDA.ipynb](EDA.ipynb) for the exploratory analysis.
2. Open and run [evaluation.ipynb](evaluation.ipynb) for training, evaluation, and model comparison. Results (metrics, classification reports, confusion matrices, predictions) are saved in the `results/` folder.
