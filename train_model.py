import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── Load clean data ───────────────────────────────────────────────────
df = pd.read_csv("data/processed/credit_clean.csv")

# ── Separate features from target ────────────────────────────────────
X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

# ── Train/test split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ── Apply SMOTE to balance training data ──────────────────────────────
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# ── Helper function to evaluate any model ────────────────────────────
# A function so we don't repeat the same evaluation code twice
# We pass in the model name, the trained model, and the test data
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{'=' * 50}")
    print(f"Model: {name}")
    print(f"{'=' * 50}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(f"  True Negatives  (correct no default): {cm[0][0]}")
    print(f"  False Positives (wrong default flag) : {cm[0][1]}")
    print(f"  False Negatives (missed defaulters)  : {cm[1][0]}")
    print(f"  True Positives  (caught defaulters)  : {cm[1][1]}")

# ── Model 1: Random Forest ────────────────────────────────────────────
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)
evaluate_model("Random Forest", rf_model, X_test, y_test)

# ── Model 2: XGBoost ─────────────────────────────────────────────────
# XGBoost builds trees sequentially - each tree corrects
# the mistakes of the previous one
# eval_metric='logloss' is the loss function for binary classification
print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train_balanced, y_train_balanced)
evaluate_model("XGBoost", xgb_model, X_test, y_test)

# ── Compare both models ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("COMPARISON SUMMARY")
print("=" * 50)
print("Look at F1-score for class 1 (defaulters)")
print("Higher F1 = better at catching defaulters")

import joblib
import os

# ── Save the best model ───────────────────────────────────────────────
# joblib saves a trained Python object to a file
# This means we never have to retrain - just load the file
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, "models/credit_risk_model.pkl")
print()
print("XGBoost model saved to models/credit_risk_model.pkl")

# ── Test loading the model back ───────────────────────────────────────
# Verify it loads correctly
loaded_model = joblib.load("models/credit_risk_model.pkl")
print("Model loaded back successfully")
print("Ready to make predictions on new customers")

import numpy as np

# ── Predict on a new customer ─────────────────────────────────────────
# Let's create two example customers and see what the model predicts
# These are made-up but realistic financial profiles

print()
print("=" * 50)
print("PREDICTING ON NEW CUSTOMERS")
print("=" * 50)

# Customer 1 - High risk profile
# High credit utilization, missed payments, high debt ratio
customer_high_risk = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": 0.95,
    "age": 35,
    "NumberOfTime30-59DaysPastDueNotWorse": 3,
    "DebtRatio": 0.8,
    "MonthlyIncome": 3000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 2,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 1,
    "NumberOfDependents": 3
}])

# Customer 2 - Low risk profile
# Low credit utilization, no missed payments, low debt ratio
customer_low_risk = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": 0.10,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.2,
    "MonthlyIncome": 8000,
    "NumberOfOpenCreditLinesAndLoans": 4,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1
}])

# predict_proba gives probability of default (0 to 1)
# [0][1] means: first customer, probability of class 1 (default)
risk_high = loaded_model.predict_proba(customer_high_risk)[0][1]
risk_low = loaded_model.predict_proba(customer_low_risk)[0][1]

print(f"Customer 1 (High Risk Profile):")
print(f"  Default probability: {risk_high:.1%}")
print(f"  Decision: {'REJECT LOAN' if risk_high > 0.5 else 'APPROVE LOAN'}")
print()
print(f"Customer 2 (Low Risk Profile):")
print(f"  Default probability: {risk_low:.1%}")
print(f"  Decision: {'REJECT LOAN' if risk_low > 0.5 else 'APPROVE LOAN'}")