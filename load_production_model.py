import mlflow
import pandas as pd

# ── Load model using its registry alias ──────────────────────────────
# Notice: we are NOT pointing to a file path
# We are asking the registry: "give me whatever is tagged @production"
model_uri = "models:/credit_risk_classifier@production"

print("Loading production model from MLflow registry...")
model = mlflow.sklearn.load_model(model_uri)
print("Model loaded successfully!")
print()

# ── Test with our high risk customer ──────────────────────────────────
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

risk = model.predict_proba(customer_high_risk)[0][1]
print(f"Default probability: {risk:.1%}")
print(f"Decision: {'REJECT LOAN' if risk > 0.5 else 'APPROVE LOAN'}")