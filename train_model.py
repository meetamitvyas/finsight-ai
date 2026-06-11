import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

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

# ── Train Random Forest model ─────────────────────────────────────────
# n_estimators=100 means 100 decision trees
# random_state=42 makes results reproducible
# n_jobs=-1 means use all CPU cores to train faster
print("Training Random Forest model...")
print("This may take 1-2 minutes...")
print()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# .fit() is the actual training step
# We pass the balanced training data
model.fit(X_train_balanced, y_train_balanced)

print("Training complete!")
print()

# ── Evaluate on test set ──────────────────────────────────────────────
# .predict() makes predictions on data the model has never seen
y_pred = model.predict(X_test)

# Classification report shows precision, recall, F1 for each class
print("Model Performance on Test Set:")
print("=" * 50)
print(classification_report(y_test, y_pred))

# Confusion matrix shows correct vs incorrect predictions
print("Confusion Matrix:")
print("=" * 50)
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives  (correctly predicted no default): {cm[0][0]}")
print(f"False Positives (predicted default, actually no): {cm[0][1]}")
print(f"False Negatives (predicted no default, actually defaulted): {cm[1][0]}")
print(f"True Positives  (correctly predicted default): {cm[1][1]}")