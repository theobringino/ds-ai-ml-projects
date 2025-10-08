import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
# Note: I am still using Ridge here even if the results are the same with that of the baseline lr just for advanced lr implementation purposes

# Declare constants
data_path = 'cyber_data.csv'
artifacts_dir = 'models'
target_var = 'Log_Local_Infection'


# This function preprocesses the data by applying the transforms and required feature engineering
def preprocess_data(df, fit_scaler=False, scaler=None):
    # Apply Log transforms to features and target var
    df['Log_Spam'] = np.log1p(df['Spam'])
    df['Log_Ransomware'] = np.log1p(df['Ransomware'])
    df['Log_Local_Infection'] = np.log1p(df['Local_Infection'])
    df['Log_Exploit'] = np.log1p(df['Exploit'])
    df['Log_Malicious_Mail'] = np.log1p(df['Malicious_Mail'])
    df['Log_Network_Attack'] = np.log1p(df['Network_Attack'])
    df['Log_Web_Threats'] = np.log1p(df['Web_Threats'])

    # Encode country
    df = pd.get_dummies(df, columns=['Country'], drop_first=True, dtype=int)
    
    # Define features to scale (all numeric features except the target)
    features_to_scale = [col for col in df.columns if col not in [target_var, 'Country']]
    
    if fit_scaler:
        scaler = StandardScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        return df, scaler
    else:
        # Use the loaded scaler for transformation
        df[features_to_scale] = scaler.transform(df[features_to_scale])
        return df, None
    
# Model training logic
def train_model():
    # Load and prepare data
    data = pd.read_csv(data_path)
    
    # Train test split
    X = data.drop(columns=[target_var])
    y = data[target_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess and fit scaler on the training data
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df_scaled, scaler = preprocess_data(train_df, fit_scaler=True)
    
    # Isolate scaled features and target
    X_train_scaled = train_df_scaled.drop(columns=[target_var])
    y_train = train_df_scaled[target_var]

    # Train the model
    model = Ridge(alpha=0.21544346900318823) # Optimal parameter from Advanced LR notebook for Ridge
    model.fit(X_train_scaled, y_train)

    # Save artifacts to models directory
    import os
    os.makedirs(artifacts_dir, exist_ok=True)
    
    joblib.dump(model, f'{artifacts_dir}/model.pkl')
    joblib.dump(scaler, f'{artifacts_dir}/scaler.pkl')
    
    print("Training complete. Model and Scaler artifacts saved to 'models/' directory.")

if __name__ == '__main__':
    train_model()