import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("credit_fraud.csv")

print("Class distribution:")
print(df["is_fraud"].value_counts())

# -----------------------
# Features / target
# -----------------------
X = df.drop(columns=["is_fraud", "transaction_id"])
y = df["is_fraud"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# -----------------------
# Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# Scaling (ONLY needed for Logistic Regression)
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 1. LOGISTIC REGRESSION
# =========================================================
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\n================ LOGISTIC REGRESSION ================")
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

print("ROC AUC:", roc_auc_score(y_test, lr_proba))


# =========================================================
# 2. RANDOM FOREST
# =========================================================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)  # no scaling needed

rf_proba = rf_model.predict_proba(X_test)[:, 1]

# default threshold (0.5)
rf_pred = (rf_proba > 0.5).astype(int)

print("\n================ RANDOM FOREST ================")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("ROC AUC:", roc_auc_score(y_test, rf_proba))