# Global Cyber Threat Landscape: E2E Linear Regression Implementation to MLOps Production Pipeline

## Project Overview
This project applies Machine Learning Operations (MLOps) principles to a predictive model for cyber risk. The model, a **Ridge Regression** developed in earlier weeks, is refactored from a Jupyter Notebook into a production-ready CI/CD pipeline.

The goal of Week 9 was to implement **Continuous Integration (CI)** via `train.py` and **Continuous Deployment (CD)** via `api.py`, and then secure the resulting API endpoint.

## MLOps Pipeline & Execution

The project follows a standard MLOps two-step process: **Build** (Training) and **Serve** (Deployment).

### Step 1: Build Phase (`train.py`) - Continuous Integration (CI)
This script handles data loading, feature engineering (log transforms, One-Hot Encoding), scaling (`StandardScaler`), model training (`Ridge`), and **artifact creation**.

**Action:** Run the training script to generate the model artifacts.

`python train.py`

This outputs the `models/`  directory containing the following artifacts, which enforce Training-Serving Consistency in the API:
1. model.pkl
2. scaler.pkl
3. feature_cols.pkl

### Step 2: Serve Phase (api.py) - Continuous Deployment (CD)
This script loads the artifacts from Step 1, initializes a Flask web server, and sets up the secure /predict endpoint.

**Action**: Run the API server. This process is continuous.

`python api.py`

#### API Security and Testing:
The `/predict endpoint` is secured with API Key Authentication to prevent unauthorized access, demonstrating the Fail-Secure Principle. While this is hardcoded as it just a simple project. It is important to note that this should not be practiced and secrets are to be stored in Secrets Managers to prevent security issues. Here are the core principles for security that were considered and how it was implemented:
- Authentication:  The API requires the custom header `X-API-Key` with a valid value.
- Fail-Secure: If the key is missing or invalid, the API returns an HTTP 401 Unauthorized response before executing any prediction logic.

### Step 3: Testing the Secure Endpoint
The test_api.py script acts as an authorized client. It sends a sample data point along with the required key.

The API Key is 23Theo23APIKey. The test script must include this in the request headers as noted by:

headers = {
    "Content-Type": "application/json",
    "X-API-Key": "23Theo23APIKey" # This header is required for authentication
}

**Action**: Run the test script (ensure api.py is running first).
`python test_api.py`

#### Expected Output: A successful 200 OK response showing the predicted infection rates.

