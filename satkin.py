import numpy as np
import cudf as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cuml import preprocessing
from cuml.ensemble import RandomForestRegressor
from cuml.metrics import r2_score
import opendatasets as od

# Download the dataset
od.download('https://www.kaggle.com/datasets/idawoodjee/predict-the-positions-and-speeds-of-600-satellites')

# Load the data
train = pd.read_csv('predict-the-positions-and-speeds-of-600-satellites/jan_train.csv')
test = pd.read_csv('predict-the-positions-and-speeds-of-600-satellites/jan_test.csv')
key = pd.read_csv('predict-the-positions-and-speeds-of-600-satellites/answer_key.csv')

# Inspect the training data
train.info()
train.epoch = pd.to_datetime(train.epoch)

# Prepare features and target variables
X = train[['id', 'sat_id', 'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']].copy().astype(np.float32)
y = train[['x', 'y', 'z', 'Vx', 'Vy', 'Vz']].astype(np.float32)

# Scale the features
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

# Separate targets
Y_train = y

# Train regressors
nstreams = 25
regressors = []
for i in range(6):
    max_features = 3 if i < 4 else 4  # Different max_features for last two regressors
    regressor = RandomForestRegressor(n_estimators=200, max_features=max_features, n_streams=nstreams)
    
    try:
        regressor.fit(X_scale, Y_train.iloc[:, i])
        regressors.append(regressor)
    except Exception as e:
        print(f"Error training regressor {i}: {e}")
        del regressor  # Explicitly delete the regressor to free memory

# Preprocess test data
test = test[['id', 'sat_id', 'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']].copy().astype(np.float32)
X_test = min_max_scaler.fit_transform(test)

# Make predictions
y_preds = [regressor.predict(X_test) for regressor in regressors]

# Calculate R-squared values and overall R-squared
r2_scores = [r2_score(key.iloc[:, i].astype(np.float32), y_preds[i]) for i in range(6)]
overall_r2 = np.mean(r2_scores)

# Print R-squared values
print("R-squared values:", r2_scores)
print("Overall R-squared:", overall_r2)

# Create a bar plot for R-squared values
plt.figure(figsize=(10, 6))
plt.bar(range(1, 7), r2_scores, color='skyblue')
plt.xticks(range(1, 7), [f'Predictor {i+1}' for i in range(6)])
plt.ylabel('R-squared Value')
plt.title('R-squared Values for Each Predictor')
plt.savefig('r_squared_plot.png')  # Save the R-squared plot
plt.show()  # Show the R-squared plot
