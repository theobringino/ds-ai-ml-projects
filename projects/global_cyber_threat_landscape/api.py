import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

artifacts_dir = 'models'
target_var = 'Log_Local_Infection'

# List of original feature columns that will be log-transformed and then dropped
ORIGINAL_FEATURES = [
    'Spam', 'Ransomware', 'Exploit', 'Malicious_Mail', 
    'Network_Attack', 'Web_Threat'
]

# --- 1. Load Artifacts ---
MODEL = None
SCALER = None
FEATURE_COLS = [] # This list is CRITICAL for aligning OHE features

try:
    MODEL = joblib.load(f'{artifacts_dir}/model.pkl')
    SCALER = joblib.load(f'{artifacts_dir}/scaler.pkl')
    # Load the feature list created during training (e.g., ['Log_Spam', ..., 'Country_US'])
    FEATURE_COLS = joblib.load(f'{artifacts_dir}/feature_cols.pkl') 
    print("Model, Scaler, and Feature List loaded successfully.")
except Exception as e:
    print(f"Error loading model artifacts from '{artifacts_dir}': {e}")
    MODEL = None
    SCALER = None


# --- 2. Preprocessing Function ---
def preprocess_input(df, scaler, feature_cols):
    """Applies all transformations to a single incoming DataFrame, matching train.py logic."""
    
    # 1. Apply Log transforms (Creates 'Log_X' columns)
    for col in ORIGINAL_FEATURES:
        if col in df.columns:
            df[f'Log_{col}'] = np.log1p(df[col])

    # 2. Encode country
    if 'Country' in df.columns:
        df = pd.get_dummies(df, columns=['Country'], drop_first=True, dtype=int)
    
    # 🚨 CRITICAL FIX: Drop original, untransformed features
    # This step ensures only the Log-transformed and OHE columns proceed.
    cols_to_drop = ORIGINAL_FEATURES
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 3. Align Columns with Training Data (Crucial for OHE stability)
    
    # Add any missing one-hot columns (e.g., a country not present in the API call)
    missing_cols = list(set(feature_cols) - set(df.columns))
    for col in missing_cols:
        df[col] = 0 
        
    # Select and order the columns to match the features used during training
    data_point_df = df[feature_cols] 

    # 4. Scaling
    input_scaled = scaler.transform(data_point_df)
    
    return input_scaled

# --- 3. API Setup and Endpoint ---
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL or not SCALER or not FEATURE_COLS:
        return jsonify({"error": "Model, Scaler, or Feature List not loaded. Check server logs."}), 500
        
    try:
        json_data = request.get_json(force=True)
        input_df = pd.DataFrame([json_data])
        
        input_scaled = preprocess_input(input_df, SCALER, FEATURE_COLS)
        
        # Make prediction
        log_prediction = MODEL.predict(input_scaled)[0]
        
        # Apply Inverse Transform (exp(x) - 1)
        final_prediction = np.expm1(log_prediction)
        
        return jsonify({
            'log_infection_rate_prediction': float(log_prediction),
            'local_infection_rate_prediction': float(final_prediction)
        }), 200

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)