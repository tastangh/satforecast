import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cuml import preprocessing
from cuml.ensemble import RandomForestRegressor
from cuml.metrics import r2_score, mean_squared_error, mean_absolute_error
import opendatasets as od
import cupy as cp  # Import CuPy

# Veri setini indir
od.download('https://www.kaggle.com/datasets/idawoodjee/predict-the-positions-and-speeds-of-600-satellites')

# Verileri yükle
train = pd.read_csv('predict-the-positions-and-speeds-of-600-satellites/jan_train.csv')
test = pd.read_csv('predict-the-positions-and-speeds-of-600-satellites/jan_test.csv')
key = pd.read_csv('predict-the-positions-and-speeds-of-600-satellites/answer_key.csv')

# Eğitim verilerini incele
train.info()

# Tarihleri dönüştür
train['epoch'] = pd.to_datetime(train['epoch'])

# Veri görselleştirme
sns.pairplot(train)
plt.show()  # Show the plot

# Özellikler ve hedef değişkeni ayır
X = train[['id', 'sat_id', 'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']].copy()
y = train[['x', 'y', 'z', 'Vx', 'Vy', 'Vz']]

# Verileri ölçeklendir
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

# Eğitim ve hedef veri boyutlarını kontrol et
print("Scaled feature shape:", X_scaled.shape, "Target shape:", y.shape)

# Hedef değişkenleri ayır
Y_train = [cp.asarray(y.iloc[:, i].to_numpy()) for i in range(y.shape[1])]  # Convert to CuPy arrays

# Random Forest Regresyon modellerini oluştur
nstreams = 25
regressors = [RandomForestRegressor(n_estimators=200, max_features=(3 if i < 4 else 4), n_streams=nstreams) for i in range(6)]

# Modelleri eğit
for i in range(6):
    regressors[i].fit(X_scaled, Y_train[i])

# Test verisini ön işleme tabi tut
X_test = test[['id', 'sat_id', 'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']].copy()
X_test_scaled = min_max_scaler.transform(X_test)

# Tahmin yap
y_preds = [regressor.predict(X_test_scaled) for regressor in regressors]

# R-kare değerlerini hesapla ve RMSE ile MAE'yi hesapla
r2_scores = []
rmse_scores = []
mae_scores = []
accuracy_scores = []

# Tolerans yüzdesi belirle (örneğin %5)
tolerance = 0.05  # 5%

for i in range(6):
    y_true = cp.asarray(key.iloc[:, i].to_numpy())  # Convert to CuPy arrays
    y_pred_cupy = cp.asarray(y_preds[i])  # Ensure y_preds is a CuPy array

    r2 = r2_score(y_true, y_pred_cupy)
    rmse = mean_squared_error(y_true, y_pred_cupy, squared=False)  # RMSE
    mae = mean_absolute_error(y_true, y_pred_cupy)  # MAE
    
    # Accuracy hesapla
    accurate_predictions = cp.abs(y_pred_cupy - y_true) <= (tolerance * cp.abs(y_true))
    accuracy = cp.mean(accurate_predictions) * 100  # Yüzde olarak hesapla

    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    accuracy_scores.append(accuracy)

# Ortak R-kare değerini hesapla
overall_r2 = np.mean(r2_scores)

# Sonuçları yazdır
print("R-kare değerleri:", r2_scores)
print("RMSE değerleri:", rmse_scores)
print("MAE değerleri:", mae_scores)
print("Accuracy değerleri:", accuracy_scores)  # % cinsinden
print("Genel R-kare değeri:", overall_r2)
