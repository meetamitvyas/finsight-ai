from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import pandas as pd

# ── Create the API application ────────────────────────────────────────
app = FastAPI(title="FinSight AI - Credit Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (fine for local development)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load the production model ONCE when the server starts ────────────
# We don't want to load the model on every request - that would be slow
print("Loading production model...")
model = mlflow.sklearn.load_model("models:/credit_risk_classifier@production")
print("Model loaded! API ready.")

# ── Define the shape of incoming data ──────────────────────────────────
# This is a "schema" - it tells FastAPI exactly what fields to expect
# and automatically validates incoming requests
class CustomerData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int

# ── Define a simple "health check" endpoint ────────────────────────────
# This is the simplest possible endpoint - just confirms the API is alive
@app.get("/")
def root():
    return {"status": "FinSight AI Credit Risk API is running"}

# ── Define the prediction endpoint ──────────────────────────────────────
# This is the main endpoint - it accepts customer data and returns a prediction
@app.post("/predict")
def predict(customer: CustomerData):
    # Convert the incoming data to a DataFrame
    # Note: we rename fields back to match our model's expected column names
    data = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": customer.RevolvingUtilizationOfUnsecuredLines,
        "age": customer.age,
        "NumberOfTime30-59DaysPastDueNotWorse": customer.NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio": customer.DebtRatio,
        "MonthlyIncome": customer.MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": customer.NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": customer.NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": customer.NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": customer.NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents": customer.NumberOfDependents
    }])

    # Get the probability of default
    risk_score = model.predict_proba(data)[0][1]

    # Make a decision based on the threshold
    decision = "REJECT" if risk_score > 0.5 else "APPROVE"

    return {
        "default_probability": round(float(risk_score), 4),
        "decision": decision
    }