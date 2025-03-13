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

## Live Demo
Check out the live project here: [House Price Prediction](https://house-price-prediction-nm9f.onrender.com)

## Author
Ketki Bansod
