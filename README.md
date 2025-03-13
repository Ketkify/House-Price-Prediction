# House Price Prediction Project

## Overview
This project implements a machine learning pipeline for predicting house prices using various regression models. It includes data preprocessing, feature engineering, model training, evaluation, and model blending.

## Project Structure
```
|-- housing.csv                  # Dataset file
|-- model.pkl                     # Trained best model (XGBoost)
|-- scaler.pkl                    # StandardScaler for feature scaling
|-- feature_columns.pkl            # Feature column names
|-- house_price_prediction.py      # Main script for training and evaluation
|-- README.md                      # Instructions and project details
```

## Dependencies
Before running the project, install the required dependencies by executing the following command:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost lightgbm catboost skopt tensorflow
```

## Steps to Run the Project

1. **Load the Dataset:**
   - Ensure `housing.csv` is present in the same directory.
   - The script reads and preprocesses the dataset.

2. **Preprocess Data:**
   - Handles missing values.
   - Encodes categorical features.
   - Applies feature transformations.

3. **Feature Engineering:**  
   - Creates additional features like `income_per_room`, `bedrooms_per_household`, etc.  
   - Applies log transformations to skewed data.

4. **Model Training:**  
   - Trains multiple models, including:  
     - Linear Regression  
     - Random Forest  
     - Gradient Boosting  
     - XGBoost  
     - CatBoost  
   - Saves the trained XGBoost model as `model.pkl`.  
   - Saves the `StandardScaler` as `scaler.pkl`.

5. **Model Evaluation:**  
   - Calculates MAE, RMSE, and R² for each model.  
   - Blends predictions from XGBoost, Gradient Boosting, and CatBoost.  
   - Saves the best model.

6. **Visualization:**  
   - Correlation heatmap.  
   - Box plots for feature distribution.  
   - Pair plots.  
   - Actual vs Predicted scatter plots.

## Running the Script
To execute the script, run:
```bash
python house_price_prediction.py
```

## Code
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data = pd.read_csv("housing.csv").dropna()

data_encoded = pd.get_dummies(data, drop_first=True)

# Feature Engineering
data_encoded['income_per_room'] = data_encoded['median_income'] / data_encoded['total_rooms']
data_encoded['bedrooms_per_household'] = data_encoded['total_bedrooms'] / data_encoded['households']
data_encoded['population_per_household'] = data_encoded['population'] / data_encoded['households']

X = data_encoded.drop("median_house_value", axis=1)
y = data_encoded["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=300, verbose=0, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred)
    }

# Save best model
best_model = models["XGBoost"]
with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("Best model saved as 'model.pkl'")
```

## Author
Ketki Bansod
