# Energy Efficiency Prediction

This project explores predicting heating and cooling loads of buildings using machine learning models. The goal is to understand how well we can estimate energy requirements based on building design features and to highlight trade-offs between heating and cooling efficiency.

## Project Overview

Dataset: Energy Efficiency dataset (UCI Machine Learning Repository)

Objective: Predict heating load and cooling load from 8 building features (e.g., surface area, wall area, roof area, glazing area, orientation).

Methods: Applied regression models and evaluated performance with error metrics.

Key Insight: Cooling loads were modeled more accurately than heating loads, suggesting architectural designs are naturally more optimized for cooling efficiency.

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

Building design variables provide strong predictive power for energy efficiency.

Cooling load models outperformed heating load models, reflecting real-world prioritization of cooling efficiency in building design.

Results highlight how machine learning can inform sustainable architecture and energy management strategies.
