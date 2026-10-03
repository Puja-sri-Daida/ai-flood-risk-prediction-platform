import joblib
import pandas as pd

print("Loading flood model...")

model_data = joblib.load(
    "models/flood_model.joblib"
)

model = model_data["model"]
features = model_data["features"]

print("Model loaded!")
print()
print("Expected features:")
print(features)

# Values collected automatically for Mumbai
data = {
    "rainfall_24h_mm": 0.3,
    "rainfall_3day_mm": 6.2,
    "humidity_pct": 81,
    "temperature_c": 27.5,
    "elevation_m": 8.0,
    "slope_deg": 0.26,
    "river_distance_km": 0.22,
    "soil_moisture_pct": 29.5,
    "historical_flood_index": 6.12
}

# Arrange values in exactly the model's expected order
input_data = pd.DataFrame(
    [[data[feature] for feature in features]],
    columns=features
)

print()
print("MODEL INPUT:")
print(input_data)

# Prediction
prediction = model.predict(input_data)[0]

print()
print("==============================")
print("FLOOD PREDICTION")
print("==============================")

print("Prediction:", prediction)

# Probability
if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(
        input_data
    )[0]

    print()
    print("PROBABILITIES:")

    for index, probability in enumerate(probabilities):

        print(
            "Class",
            index,
            ":",
            round(probability * 100, 2),
            "%"
        )
