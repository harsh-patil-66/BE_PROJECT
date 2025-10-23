from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

# ==============================
# Load model and feature names
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model(2).pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")

model = joblib.load(MODEL_PATH)
model_features = joblib.load(FEATURES_PATH)

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Credit Risk Prediction API is running successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # ==============================
        # Preprocessing (match training)
        # ==============================
        categorical_cols = [
            'Employment_Status', 'Industry_Sector', 'Education_Level',
            'Marital_Status', 'Housing_Status', 'Credit_Mix',
            'Loan_Purpose', 'Collateral_Type', 'Seasonal_Spending_Pattern'
        ]

        # Ensure all categorical columns exist
        for col in categorical_cols:
            if col not in df.columns:
                df[col] = "Unknown"

        # Convert boolean-like columns
        flag_cols = ["Bankruptcy_Flag", "Bankruptcy_Trigger_Flag"]
        for col in flag_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .map({
                        "true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0
                    })
                    .fillna(0)
                    .astype(int)
                )

        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Align with training features
        missing_cols = [col for col in model_features if col not in df_encoded.columns]
        for col in missing_cols:
            df_encoded[col] = 0

        # Drop extra columns
        df_encoded = df_encoded[model_features]

        # ==============================
        # Predict
        # ==============================
        y_pred = model.predict(df_encoded)
        y_pred = float(np.clip(y_pred[0], 300, 900))

        return jsonify({
            "predicted_credit_risk_score": round(y_pred, 2),
            "total_features_used": len(df_encoded.columns)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# Run Flask API
# ==============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
