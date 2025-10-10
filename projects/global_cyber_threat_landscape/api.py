import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import os

# CONSTANTS
ARTIFACTS_DIR = 'models'
TARGET_VAR_ORIG = 'Local Infection'
LOG_TRANSFORM_COLS = [
    'Spam', 'Ransomware', 'Exploit', 'Malicious Mail', 
    'Network Attack', 'Web Threat'
]
# API Key
API_KEY = "23Theo23APIKey"

# 1. Load Artifacts
MODEL = None
SCALER = None
FEATURE_COLS = [] 

try:
    MODEL = joblib.load(f'{ARTIFACTS_DIR}/model.pkl')
    SCALER = joblib.load(f'{ARTIFACTS_DIR}/scaler.pkl')
    FEATURE_COLS = joblib.load(f'{ARTIFACTS_DIR}/feature_cols.pkl') 
    print("Model, Scaler, and Feature List loaded successfully.")
except Exception as e:
    print(f"Error loading model artifacts from '{ARTIFACTS_DIR}': {e}")
    print("Please run train.py first to create the necessary files.")

# --- 2. Preprocessing Function for API Input ---
def preprocess_input(df, scaler, feature_cols):
    """
    Applies all transformations to a single incoming DataFrame, 
    matching train.py logic and ensuring column alignment.
    """
    # 1. Standardize column names
    df.columns = df.columns.str.replace(' ', '_')
    
    # 2. Apply Log transforms (Creates 'Log_X' columns)
    log_features_to_drop = []
    for col in LOG_TRANSFORM_COLS:
        col = col.replace(' ', '_')
        log_col_name = f'Log_{col}'
        if col in df.columns:
            # We assume the API input contains the original untransformed values
            df[log_col_name] = np.log1p(df[col])
            log_features_to_drop.append(col)
        
    # 3. One-Hot Encode the Country column
    df = pd.get_dummies(df, columns=['Country'], drop_first=True, dtype=int)
    
    # 4. Drop original untransformed features
    df = df.drop(columns=log_features_to_drop, errors='ignore')
    
    # Add missing one-hot columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0 
        
    # Select and order the columns to match the features used during training
    data_point_df = df[feature_cols] 

    # 5. Scaling
    input_scaled = scaler.transform(data_point_df)
    
    return input_scaled

# 3. API Setup and Endpoint
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL or not SCALER or not FEATURE_COLS:
        return jsonify({"error": "Model, Scaler, or Feature List not loaded. Please run train.py first."}), 500

    received_key = request.headers.get('X-API-Key')
    if received_key != API_KEY:
        # Implement Fail-Secure: return 401 Unauthorized and stop processing
        return jsonify({"error": "Unauthorized access. Invalid or missing API Key."}), 401


    try:
        # Get JSON data, wrap in a list to create a one-row DataFrame
        json_data = request.get_json(force=True)
        input_df = pd.DataFrame([json_data])
        
        # Apply preprocessing
        input_scaled = preprocess_input(input_df.copy(), SCALER, FEATURE_COLS)
        
        # Make prediction
        log_prediction = MODEL.predict(input_scaled)[0]
        
        # Apply Inverse Transform (exp(x) - 1)
        final_prediction = np.expm1(log_prediction)
        
        return jsonify({
            'log_infection_rate_prediction': float(log_prediction),
            'local_infection_rate_prediction': float(final_prediction) # Inverse transformed value
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed due to processing error: {e}"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)