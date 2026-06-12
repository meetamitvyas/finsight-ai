import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# ── Load reference and current data ───────────────────────────────────
# Reference = the data our model was trained on
# Current = simulated "new" data representing this month's customers
reference_data = pd.read_csv("data/processed/credit_clean.csv")
current_data = pd.read_csv("data/processed/credit_current.csv")

# We only compare the feature columns, not the target column
# (in production, we wouldn't know the target for new customers yet)
feature_columns = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
    "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"
]

reference_features = reference_data[feature_columns]
current_features = current_data[feature_columns]

# ── Create and run the drift report ───────────────────────────────────
# DataDriftPreset automatically checks every column for distribution changes
print("Running drift detection...")
report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=reference_features, current_data=current_features)

# ── Save as HTML report ────────────────────────────────────────────────
result.save_html("drift_report.html")
print("Drift report saved to drift_report.html")
print("Open this file in your browser to see the results.")