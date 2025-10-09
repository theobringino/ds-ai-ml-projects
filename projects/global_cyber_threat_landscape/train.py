import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# CONSTANTS
DATA_PATH = 'cyber_data.csv'
ARTIFACTS_DIR = 'models'
TARGET_VAR_ORIG = 'Local Infection'
TARGET_VAR_LOG = 'Log_Local_Infection'

# Columns to be log-transformed (these originals will be dropped later)
LOG_TRANSFORM_COLS = [
    'Spam', 'Ransomware', 'Exploit', 'Malicious Mail', 
    'Network Attack', 'Web Threat'
]

# Columns to drop entirely (e.g., meta data, rank columns)
COLS_TO_DROP = [
    'AttackDate', 'index', 'On Demand Scan',
    'Rank Spam', 'Rank Ransomware', 'Rank Local Infection', 
    'Rank Exploit', 'Rank Malicious Mail', 'Rank Network Attack', 
    'Rank On Demand Scan', 'Rank Web Threat'
]

# The optimal alpha found in the advanced notebook
RIDGE_ALPHA = 0.21544346900318823

# --- PREPROCESSING FUNCTION (Only handles feature engineering) ---
def preprocess_features(df):
    """
    Applies all feature engineering and cleaning steps except for scaling.
    This function is reusable in the API.
    """
    # 1. Standardize column names (replace spaces with underscores)
    df.columns = df.columns.str.replace(' ', '_')
    
    # 2. Drop unnecessary metadata/rank columns
    cols_to_drop = [col for col in COLS_TO_DROP if col in df.columns]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 3. Drop rows with any missing values
    df.dropna(inplace=True)

    # 4. Log Transform features (np.log1p)
    log_features_to_drop = []
    for col in LOG_TRANSFORM_COLS:
        col = col.replace(' ', '_') # Use the standardized name
        log_col_name = f'Log_{col}'
        df[log_col_name] = np.log1p(df[col])
        log_features_to_drop.append(col)
        
    # 5. One-Hot Encode the Country column
    df = pd.get_dummies(df, columns=['Country'], drop_first=True, dtype=int)
    
    # 6. Drop original untransformed features
    df = df.drop(columns=log_features_to_drop, errors='ignore')
    
    return df

# Training Script
def train_model():
    # 1. Setup paths
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Please ensure 'cyber_data.csv' is in the same directory.")
        return

    # 2. Apply preprocessing steps (cleaning and feature engineering)
    # Note: Target transformation is done here, as it is a feature engineering step
    processed_df = preprocess_features(data.copy())
    
    # 3. Log Transform the Target variable
    processed_df[TARGET_VAR_LOG] = np.log1p(processed_df[TARGET_VAR_ORIG.replace(' ', '_')])
    
    # 4. Define X and y (features and target)
    X = processed_df.drop(columns=[TARGET_VAR_ORIG.replace(' ', '_'), TARGET_VAR_LOG])
    y = processed_df[TARGET_VAR_LOG]
    
    # 5. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Capture the final ordered feature list for API use
    feature_cols = X_train.columns.tolist()
    
    # 7. Scaling (Fit on Train, Transform Train and Test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for consistency
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=feature_cols)
    
    # 8. Train the Ridge Model
    model = Ridge(alpha=RIDGE_ALPHA) 
    model.fit(X_train_scaled, y_train)

    # 9. Evaluate (optional, but confirms success)
    r2 = model.score(X_test_scaled, y_test)
    print(f"\n--- Training Complete ---")
    print(f"Model: Ridge Regression (Alpha={RIDGE_ALPHA})")
    print(f"Test Set R-squared: {r2:.4f}")

    # 10. Save Artifacts
    joblib.dump(model, f'{ARTIFACTS_DIR}/model.pkl')
    joblib.dump(scaler, f'{ARTIFACTS_DIR}/scaler.pkl')
    joblib.dump(feature_cols, f'{ARTIFACTS_DIR}/feature_cols.pkl')
    print(f"Model artifacts saved to '{ARTIFACTS_DIR}/'")

if __name__ == '__main__':
    train_model()