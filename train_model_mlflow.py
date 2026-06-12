import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

# ── Load clean data ───────────────────────────────────────────────────
df = pd.read_csv("data/processed/credit_clean.csv")

X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# ── Set the experiment name ───────────────────────────────────────────
# This groups all our runs under one experiment in the MLflow UI
mlflow.set_experiment("credit_risk_model")

# ── Define the hyperparameters for this run ───────────────────────────
# Hyperparameters = settings we choose BEFORE training
# n_estimators = number of trees XGBoost builds
# max_depth = how deep each tree can grow
# learning_rate = how much each tree corrects the previous one's mistakes
"""params = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": 42
}"""
params = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05,
    "random_state": 42
}
# ── Start an MLflow run ────────────────────────────────────────────────
# Everything inside "with mlflow.start_run()" gets tracked automatically
# with mlflow.start_run(run_name="xgboost_baseline"):
with mlflow.start_run(run_name="xgboost_deeper_trees"):

    # Log the parameters - so we remember what settings we used
    mlflow.log_params(params)

    # Train the model with these parameters
    print("Training XGBoost with MLflow tracking...")
    model = XGBClassifier(**params, n_jobs=-1, eval_metric='logloss')
    model.fit(X_train_balanced, y_train_balanced)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    f1 = f1_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)

    # Log the metrics - so we remember the results
    mlflow.log_metric("f1_score_defaulters", f1)
    mlflow.log_metric("recall_defaulters", recall)

    # Log the model itself - saved as an MLflow artifact
    mlflow.sklearn.log_model(model, name="model")

    print(f"\nRun complete!")
    print(f"F1 Score (defaulters): {f1:.3f}")
    print(f"Recall (defaulters): {recall:.3f}")
    print(f"\nCheck MLflow UI at http://127.0.0.1:5000")