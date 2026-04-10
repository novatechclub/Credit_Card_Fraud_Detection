import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("credit_fraud.csv")

# Convert text column into numeric columns
df = pd.get_dummies(df, columns=["merchant_category"], drop_first=True)

# Features and target
X = df.drop(["is_fraud", "transaction_id"], axis=1)
y = df["is_fraud"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


print("First rows:")
print(df.head())

print("\nClass distribution:")
print(df["is_fraud"].value_counts())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("credit_fraud.csv")

X = df.drop(columns=["is_fraud", "transaction_id"])
y = df["is_fraud"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

print(model.score(X_test, y_test))