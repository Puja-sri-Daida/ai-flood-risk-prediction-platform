import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier


# ============================================================
# 1. Load dataset
# ============================================================

DATA_PATH = "data/flood_demo_dataset.csv"

FEATURES = [
    "rainfall_24h_mm",
    "rainfall_3day_mm",
    "humidity_pct",
    "temperature_c",
    "elevation_m",
    "slope_deg",
    "river_distance_km",
    "soil_moisture_pct",
    "historical_flood_index",
]

TARGET = "flood"

data = pd.read_csv(DATA_PATH)

print("\n" + "=" * 60)
print("FLOOD MODEL EVALUATION")
print("=" * 60)

print(f"\nDataset rows: {len(data)}")
print(f"Features: {len(FEATURES)}")

# ============================================================
# 2. Prepare X and y
# ============================================================

X = data[FEATURES]
y = data[TARGET]

print("\nClass distribution:")
print(y.value_counts())


# ============================================================
# 3. Create stratified train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTrain samples:", len(X_train))
print("Test samples :", len(X_test))


# ============================================================
# 4. Recreate the current XGBoost model
# ============================================================

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
    enable_categorical=True,
)


# ============================================================
# 5. Train only on training data
# ============================================================

print("\nTraining evaluation model...")

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 6. Predict unseen test data
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. Calculate evaluation metrics
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)


# ============================================================
# 8. Display results
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"\nAccuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")


# ============================================================
# 9. Confusion matrix
# ============================================================

print("\n" + "-" * 60)
print("CONFUSION MATRIX")
print("-" * 60)

print("\n                 Predicted")
print("                 No Flood   Flood")
print(f"Actual No Flood    {cm[0][0]:4d}      {cm[0][1]:4d}")
print(f"Actual Flood       {cm[1][0]:4d}      {cm[1][1]:4d}")


# ============================================================
# 10. Detailed classification report
# ============================================================

print("\n" + "-" * 60)
print("CLASSIFICATION REPORT")
print("-" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Flood", "Flood"],
        zero_division=0,
    )
)

print("=" * 60)
print("Evaluation completed successfully.")
print("=" * 60)
