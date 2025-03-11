import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and feature columns
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Remove 'median_house_value' if present
if "median_house_value" in feature_columns:
    feature_columns.remove("median_house_value")

# Save the corrected feature columns
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("✅ Fixed feature_columns.pkl! Removed 'median_house_value'.")

# Load the scaler used in training
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html", prediction=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("📩 Raw Form Data:", request.form)
        data = request.form.to_dict()

        if not data:
            error_message = "⚠️ No data received. Please fill out the form."
            print(error_message)
            return render_template("index.html", prediction=None, error=error_message)

        required_fields = [col for col in feature_columns if col not in ["bedroom_ratio", "household_rooms"] and col not in ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]]
        missing_fields = [field for field in required_fields if field not in data or data[field].strip() == ""]

        if missing_fields:
            error_message = f"⚠️ Missing fields: {', '.join(missing_fields)}"
            print(error_message)
            return render_template("index.html", prediction=None, error=error_message)

        ocean_proximity_mapping = {
            "<1H OCEAN": [1, 0, 0, 0, 0],
            "INLAND": [0, 1, 0, 0, 0],
            "ISLAND": [0, 0, 1, 0, 0],
            "NEAR BAY": [0, 0, 0, 1, 0],
            "NEAR OCEAN": [0, 0, 0, 0, 1]
        }

        try:
            numerical_features = {key: float(data[key]) for key in ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]}
        except ValueError as ve:
            print(f"❌ ValueError: {ve}")
            return render_template("index.html", prediction=None, error="Invalid numerical input. Please enter valid numbers.")

        numerical_features["bedroom_ratio"] = numerical_features["total_bedrooms"] / numerical_features["total_rooms"] if numerical_features["total_rooms"] > 0 else 0
        numerical_features["household_rooms"] = numerical_features["total_rooms"] / numerical_features["households"] if numerical_features["households"] > 0 else 0

        for key in ["total_rooms", "total_bedrooms", "population", "households"]:
            if numerical_features[key] > 0:
                numerical_features[key] = np.log(numerical_features[key] + 1)

        ocean_proximity = data.get("ocean_proximity", "INLAND")
        encoded_ocean_proximity = ocean_proximity_mapping.get(ocean_proximity, [0, 1, 0, 0, 0])

        input_df = pd.DataFrame([{**numerical_features, **dict(zip(["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"], encoded_ocean_proximity))}], columns=feature_columns)
        input_features = scaler.transform(input_df)

        prediction = model.predict(input_features)[0]
        print(f"🎯 Predicted Price: ${prediction:.2f}")

        return render_template("index.html", prediction=round(prediction, 2), error=None)

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        return render_template("index.html", prediction=None, error=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)