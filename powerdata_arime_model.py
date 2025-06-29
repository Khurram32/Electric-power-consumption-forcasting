# Import common libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load dataset
df = pd.read_csv(r'C:\vscode\python-workspace\power_data.csv')  # Replace with your file path
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Resample to daily consumption (if needed)
df = df.resample('D').mean().dropna()

# Feature engineering
df['Day'] = df.index.dayofyear
df['Weekday'] = df.index.weekday
df['Month'] = df.index.month

# Assume Power_Consumption is the target
y = df['Power_Consumption']
X = df[['Day', 'Weekday', 'Month']]

# Train/test split for ML models
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize machine learning models
ml_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'SVM': SVR(),
    'k-NN': KNeighborsRegressor(n_neighbors=5)
}

ml_results = {}

# Train & evaluate ML models
for name, model in ml_models.items():
    if name in ['SVM', 'k-NN']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    ml_results[name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    }

    plt.figure(figsize=(10, 4))
    plt.plot(y_test.values[:50], label='Actual', color='black')
    plt.plot(y_pred[:50], label=f'{name} Prediction')
    plt.title(f'Actual vs Predicted - {name}')
    plt.xlabel('Sample Index')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.show()

# ARIMA Model
arima_order = (5, 1, 0)  # Change as per AIC/BIC analysis
arima_model = ARIMA(y_train, order=arima_order)
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(y_test))

# SARIMA Model
sarima_order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
sarima_model = SARIMAX(y_train, order=sarima_order, seasonal_order=seasonal_order)
sarima_fit = sarima_model.fit()
sarima_forecast = sarima_fit.forecast(steps=len(y_test))

# ARIMA Metrics
ml_results["ARIMA"] = {
    'MAE': mean_absolute_error(y_test, arima_forecast),
    'RMSE': np.sqrt(mean_squared_error(y_test, arima_forecast)),
    'R²': r2_score(y_test, arima_forecast)
}

# SARIMA Metrics
ml_results["SARIMA"] = {
    'MAE': mean_absolute_error(y_test, sarima_forecast),
    'RMSE': np.sqrt(mean_squared_error(y_test, sarima_forecast)),
    'R²': r2_score(y_test, sarima_forecast)
}

# Plot ARIMA
plt.figure(figsize=(10, 4))
plt.plot(y_test.values[:50], label='Actual', color='black')
plt.plot(arima_forecast[:50], label='ARIMA Forecast')
plt.title('ARIMA - Actual vs Forecast')
plt.legend()
plt.show()

# Plot SARIMA
plt.figure(figsize=(10, 4))
plt.plot(y_test.values[:50], label='Actual', color='black')
plt.plot(sarima_forecast[:50], label='SARIMA Forecast')
plt.title('SARIMA - Actual vs Forecast')
plt.legend()
plt.show()

# Print all results
print("\n--- Model Evaluation Summary ---")
for model, metrics in ml_results.items():
    print(f"\n{model}:")
    for m, val in metrics.items():
        print(f"  {m}: {val:.4f}")