import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

artifacts_dir = 'models'
target_var = 'Log_Local_Infection'

# Load the saved model and scaler
try:
    MODEL = joblib.load(f'{artifacts_dir}/model.pkl')
    SCALER = joblib.load(f'{artifacts_dir}/scaler.pkl')
    # List of features expected by the model (used for re-indexing)
    # IMPORTANT: You must get the full list of features used in X_train_scaled
    # e.g., features = ['Log_Spam', 'Log_Web_Threats', 'Country_US', 'Country_CN', ...]
    # For now, we'll make a simplifying assumption:
    feature_cols = ['Log_Spam', 'Log_Web_Threats',  'Country_US', 'Country_CN'] ### Incomplete list, to finalize
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    MODEL = None
    SCALER = None


def preprocess_input(df, scaler, feature_cols):
    """Applies all transformations to a single incoming DataFrame."""
    
    # 1. Apply Log transforms (Must match train.py)
    df['Log_Spam'] = np.log1p(df['Spam'])
    df['Log_Ransomware'] = np.log1p(df['Ransomware'])
    df['Log_Local_Infection'] = np.log1p(df['Local_Infection'])
    df['Log_Exploit'] = np.log1p(df['Exploit'])
    df['Log_Malicious_Mail'] = np.log1p(df['Malicious_Mail'])
    df['Log_Network_Attack'] = np.log1p(df['Network_Attack'])
    df['Log_Web_Threats'] = np.log1p(df['Web_Threats'])

    # Categorical encoding for Country
    data_point_df = pd.get_dummies(df, columns=['Country'], drop_first=True, dtype=int)

    # Align Columns with Training Data

    missing_cols = list(set(feature_cols) - set(data_point_df.columns))
    for col in missing_cols:
        data_point_df[col] = 0
    data_point_df = data_point_df[feature_cols] 

    # Scaling
    data_point_df = scaler.transform(data_point_df)
    
    return data_point_df

# --- 3. API Setup and Endpoint ---
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL or not SCALER:
        return jsonify({"error": "Model or Scaler not loaded. Check server logs."}), 500
        
    try:
        # Get data from POST request
        json_data = request.get_json(force=True)
        
        # Convert JSON data to a pandas DataFrame
        input_df = pd.DataFrame([json_data])
        
        # Preprocess and Scale the input
        input_scaled = preprocess_input(input_df, SCALER, feature_cols)
        
        # Make prediction
        log_prediction = MODEL.predict(input_scaled)[0]
        
        # Apply Inverse Transform 
        final_prediction = np.expm1(log_prediction)
        

        return jsonify({
            'log_infection_rate_prediction': float(log_prediction),
            'local_infection_rate_prediction': float(final_prediction)
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)