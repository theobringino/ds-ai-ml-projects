import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Declare constants
data_path = 'cyber_data.csv'
artifacts_dir = 'models'
target_var = 'Log_Local_Infection'

# List of original features that will be log-transformed and then DROPPED
ORIGINAL_FEATURES_TO_DROP = [
    'Spam', 'Ransomware', 'Exploit', 'Malicious_Mail', 
    'Network_Attack', 'Web_Threat'
]


# This function preprocesses the data by applying the transforms and required feature engineering
def preprocess_data(df, fit_scaler=False, scaler=None):
    
    # 1. Apply Log transforms to features (creates Log_X columns)
    for col in ORIGINAL_FEATURES_TO_DROP:
        df[f'Log_{col}'] = np.log1p(df[col])

    # 2. Encode country (drops the original 'Country' column)
    df = pd.get_dummies(df, columns=['Country'], drop_first=True, dtype=int)
    
    # 3. Drop ORIGINAL untransformed features that were replaced by Log_X
    # 3. Drop original, untransformed features and the original un-logged target
    cols_to_drop = ORIGINAL_FEATURES_TO_DROP + ['Local_Infection']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # 4. Define features to scale (all remaining columns except the target)
    features_to_scale = [col for col in df.columns if col not in [target_var]]

    
    # 4. Drop the target column to isolate features for scaling if it exists
    cols_to_drop = [target_var, 'Local_Infection']
    features_df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    features_to_scale = features_df.columns.tolist()

    if fit_scaler:
        scaler = StandardScaler()
        # Scale the defined features
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        return df, scaler
    else:
        # Use the loaded scaler for transformation
        df[features_to_scale] = scaler.transform(df[features_to_scale])
        return df, None
    
# Model training logic
def train_model():
     #List of columns to EXCLUDE from the feature set
    COLS_TO_EXCLUDE = [
    target_var, 'Local_Infection', 'AttackDate', 
    'On_Demand_Scan', # Exclude this raw feature
    'Rank_Spam', 'Rank_Ransomware', 'Rank_Local_Infection', 
    'Rank_Exploit', 'Rank_Malicious_Mail', 'Rank_Network_Attack', 
    'Rank_On_Demand_Scan', 'Rank_Web_Threat' # Exclude all rank features
    ]
    # 1. Load and prepare data
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.replace(' ', '_', regex=True)
    data.dropna(inplace=True) 
    
    # FIX 1: Drop the date/time column
    data = data.drop(columns=['AttackDate'], errors='ignore')
    
    # 2. Create log-transformed target variable
    data[target_var] = np.log1p(data['Local_Infection'])

    # 3. Train test split
    X = data.drop(columns=[col for col in COLS_TO_EXCLUDE if col in data.columns], errors='ignore')
    y = data[target_var]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Preprocess and fit scaler on the training data
    # Pass the entire data split to preprocess_data
    train_df = X_train 
    train_df = pd.concat([X_train, y_train], axis=1)
    
    train_df_scaled, scaler = preprocess_data(train_df, fit_scaler=True)
    
    # 5. Isolate scaled features and target
    X_train_scaled = train_df_scaled.drop(columns=[target_var])
    y_train_scaled = train_df_scaled[target_var]

    # 6. Capture the final feature list for use in api.py
    feature_cols = X_train_scaled.columns.tolist()
    
    # 7. Train the model
    model = Ridge(alpha=0.21544346900318823) 
    model.fit(X_train_scaled, y_train_scaled)

    # 8. Save artifacts to models directory
    os.makedirs(artifacts_dir, exist_ok=True)
    
    joblib.dump(model, f'{artifacts_dir}/model.pkl')
    joblib.dump(scaler, f'{artifacts_dir}/scaler.pkl')
    # Save the feature column list, which is CRITICAL for API stability
    joblib.dump(feature_cols, f'{artifacts_dir}/feature_cols.pkl') 
    
    print("Training complete. Model, Scaler, and Feature List artifacts saved to 'models/' directory.")

if __name__ == '__main__':
    train_model()