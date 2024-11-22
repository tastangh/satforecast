# Predicting Satellite Positions and Velocities

## Overview
This project focuses on predicting the positions (`x`, `y`, `z`) and velocities (`Vx`, `Vy`, `Vz`) of satellites based on simulated input data. By utilizing GPU-accelerated Random Forest models, the aim is to achieve accurate predictions and evaluate performance using R-squared metrics.

---

## Steps

### Dataset
The dataset includes the following files:
- `jan_train.csv`: Contains training data with simulated and actual satellite positions and velocities.
- `jan_test.csv`: Test data including only simulated values.
- `answer_key.csv`: The ground truth data for evaluating model predictions.

### Data Preparation
- Converted the `epoch` column in the training data to datetime for potential use in feature engineering.
- Selected features for the model:
  - Identifiers: `id`, `sat_id`
  - Simulated position and velocity data: `x_sim`, `y_sim`, `z_sim`, `Vx_sim`, `Vy_sim`, `Vz_sim`
- Applied MinMax scaling to normalize the input features for better compatibility with the regression model.

### Modeling Approach
- Trained six separate Random Forest models, one for each target variable (`x`, `y`, `z`, `Vx`, `Vy`, `Vz`).
- Configured the models with the following hyperparameters:
  - 200 estimators in the forest for each model.
  - Variable `max_features` parameter to optimize performance:
    - First four targets (`x`, `y`, `z`, `Vx`): Max features set to 3.
    - Last two targets (`Vy`, `Vz`): Max features set to 4.
  - Used 25 parallel streams for faster computation.

### Prediction and Evaluation
- Predicted all six target variables for the test set using the trained models.
- Computed R-squared scores for each variable by comparing predictions with the ground truth from `answer_key.csv`.
- Calculated the average R-squared score to represent overall model performance.

### Visualization
Generated a bar chart of R-squared scores for all predictors to visualize model performance. The chart is saved as `r_squared_plot.png`.

---

## Results
- Displayed R-squared values for each target variable, showing how well the models performed.
- Included an overall R-squared score to summarize predictive accuracy across all target variables.

---

## Prerequisites
The following libraries are required to run the project:
- `numpy`
- `cudf` (GPU-accelerated DataFrame operations)
- `seaborn`
- `matplotlib`
- `cuml` (GPU-accelerated machine learning)
- `opendatasets` (for downloading datasets from Kaggle)

---

## How to Run
1. Install the required dependencies.
2. Ensure your system has GPU compatibility with CUDA for using `cuml` and `cudf`.
3. Execute the script to preprocess the data, train models, and generate predictions.
4. View the output including R-squared scores and a visualization of model performance.

---

## Outputs
- **R-squared Metrics**: Detailed scores for each target variable.
- **Overall Performance**: Average R-squared score for all predictions.
- **Visualization**: Bar chart saved as `r_squared_plot.png` to highlight model performance.
