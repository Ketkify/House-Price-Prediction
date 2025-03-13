import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# ✅ Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Load feature column names
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ✅ Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ✅ Load MAE from evaluation (Set this from your best model training)
mae_blend = 30557.40  # Use actual MAE from the blended model

@app.route("/")
def home():
    return render_template("index.html", prediction=None, error=None, price_range=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("📩 Received Form Data:", request.form)
        data = request.form.to_dict()

        if not data:
            error_message = "⚠️ No data received. Please fill out the form."
            print(error_message)
            return render_template("index.html", prediction=None, error=error_message, price_range=None)

        # ✅ Convert user input to float (excluding ocean_proximity)
        numerical_features = [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income"
        ]

        try:
            for key in numerical_features:
                data[key] = float(data[key])
        except ValueError as ve:
            print(f"❌ ValueError: {ve}")
            return render_template("index.html", prediction=None, error="Invalid numerical input. Please enter valid numbers.", price_range=None)

        # ✅ Feature Engineering (Ensure all required fields are present)
        data["income_per_room"] = data["median_income"] / data["total_rooms"]
        data["bedrooms_per_household"] = data["total_bedrooms"] / data["households"]
        data["population_per_household"] = data["population"] / data["households"]

        # ✅ Log transformation for skewed features (consistent with training)
        for col in ["total_rooms", "total_bedrooms", "population", "households"]:
            if data[col] > 0:
                data[col] = np.log1p(data[col])

        # ✅ One-Hot Encode ocean_proximity
        ocean_proximity_categories = ["INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
        ocean_proximity = data.get("ocean_proximity", "INLAND")

        # One-hot encode the selected category
        for category in ocean_proximity_categories:
            data[f"ocean_proximity_{category}"] = 1 if ocean_proximity == category else 0
        
        # Remove ocean_proximity as it's now encoded
        del data["ocean_proximity"]

        # ✅ Convert data to a DataFrame
        input_df = pd.DataFrame([data])

        # ✅ Ensure column order matches the trained model
        for feature in feature_columns:
            if feature not in input_df:
                input_df[feature] = 0  # Fill missing one-hot encoded categories

        input_df = input_df[feature_columns]  # Ensure correct feature order

        # ✅ Scale input data
        input_scaled = scaler.transform(input_df)

        # ✅ Make Prediction
        predicted_price = model.predict(input_scaled)[0]

        # ✅ Compute the price range based on MAE
        lower_bound = predicted_price - mae_blend
        upper_bound = predicted_price + mae_blend

        print(f"🎯 Predicted Price: ${predicted_price:,.2f}")
        print(f"🔹 Price Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")

        return render_template(
            "index.html",
            prediction=f"${predicted_price:,.2f}",
            price_range=f"${lower_bound:,.2f} - ${upper_bound:,.2f}",
            error=None
        )

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        return render_template("index.html", prediction=None, error=f"Unexpected error: {str(e)}", price_range=None)

if __name__ == "__main__":
    app.run(debug=True)