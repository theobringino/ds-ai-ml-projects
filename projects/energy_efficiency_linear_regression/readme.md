# Energy Efficiency Prediction (UC Irvine Dataset)

This project explores the UCI Energy Efficiency Dataset, focusing on predicting Heating Load and Cooling Load of residential buildings. Using Exploratory Data Analysis (EDA) and Linear Regression, the project investigates whether training separate models or a multi-output regression model is more effective.

## Project Overview

Dataset: Energy Efficiency dataset (UCI Machine Learning Repository)

Objective: Predict heating load and cooling load from 8 building features (e.g., surface area, wall area, roof area, glazing area, orientation).

Methods: Applied regression models and evaluated performance with error metrics.
- Approach 1: Train two separate Linear Regression models for Heating Load and Cooling Load.
- Approach 2: Train one multi-output Linear Regression model for both Heating and Cooling loads simultaneously.

Key Insight: Both approaches highlight energy-efficient building designs, with cooling load predictions performing slightly better than heating load.Cooling loads were modeled more accurately than heating loads, suggesting architectural designs are naturally more optimized for cooling efficiency.

## Methods

Data preprocessing: Cleaned dataset, handled features, and scaled inputs.

Modeling: Trained regression models (starting with Linear Regression, extended to others if needed).

Evaluation: Used
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² (Coefficient of Determination)

## Results

Heating Load: R² ≈ 0.88 → high accuracy but slightly more variance
Cooling Load: R² ≈ 0.91 → very strong predictive performance

Interpretation:
- Cooling efficiency is easier to model due to stronger correlations between design features and energy demand.
- Heating loads are influenced by additional factors (e.g., insulation, humidity, materials) not fully captured in the dataset.

## Key Takeaways

- Building design variables provide strong predictive power for energy efficiency.
- Cooling load models outperformed heating load models, reflecting real-world prioritization of cooling efficiency in building design.
- Results highlight how machine learning can inform sustainable architecture and energy management strategies.
- Heating and Cooling loads can be predicted with Linear Regression.
    - Approach 1: Separate models provide clear interpretability for each target.
    - Approach 2: A multi-output model handles both targets in one go, with comparable performance.
Final Takeaway: Buildings are generally energy-efficient, leaning more towards cooling load efficiency than heating.

## How to run
1. Clone the repository
    Run the following:
    - git clone https://github.com/your-username/energy-efficiency.git
    - cd energy-efficiency
2. Install required packages, you need to get this from the root folder of the repo as this is a shared requirements.txt file.
    Run the following:
    - pip install -r requirements.txt
3. Open the Jupyter notebooks and run the cells.

