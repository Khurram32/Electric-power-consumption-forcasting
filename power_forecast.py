import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import os

# Path
file_path = r'C:\vscode\python-workspace\power_data.csv'

# Load or create sample data
if not os.path.exists(file_path):
    print("CSV not found. Creating sample dataset...")
    dates = pd.date_range(start='2025-01-01', periods=100)
    consumption = np.random.randint(100, 200, size=100)
    sample_df = pd.DataFrame({'Date': dates, 'Power_Consumption': consumption})
    sample_df.to_csv(file_path, index=False)
    print("Sample dataset created at:", file_path)

# Load data
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

X = df[['Days']]
y = df['Power_Consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Models dictionary
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'k-NN': KNeighborsRegressor(n_neighbors=5),
    'SVM': SVR(kernel='rbf')
}

predictions = {}
forecast_results = {}
residuals = {}

future_days = np.arange(df['Days'].max() + 1, df['Days'].max() + 31).reshape(-1, 1)
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)

# Train and predict
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X)
    future_pred = model.predict(future_days)
    
    predictions[name] = pred
    forecast_results[name] = future_pred
    residuals[name] = y.values - pred

# Add predictions to DataFrame
for name in predictions:
    df[f'{name}_Pred'] = predictions[name]

# ========== Plot 1: Actual vs All Models ==========
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], y, label='Actual', color='black')
for name in models:
    plt.plot(df['Date'], df[f'{name}_Pred'], label=name, linestyle='--')
plt.title('Actual vs Model Predictions')
plt.xlabel('Date')
plt.ylabel('Power Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== Plot 2: Forecast Comparison ==========
plt.figure(figsize=(14, 6))
for name in forecast_results:
    plt.plot(future_dates, forecast_results[name], label=name, marker='o')
plt.title('Forecast Comparison (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Forecasted Consumption (kWh)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== Plot 3: Residuals ==========
plt.figure(figsize=(14, 6))
for name in residuals:
    plt.plot(df['Date'], residuals[name], label=f'{name} Residuals')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals (Error) for Each Model')
plt.xlabel('Date')
plt.ylabel('Residual (Actual - Predicted)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== Forecast Table ==========
forecast_df = pd.DataFrame({'Date': future_dates})
for name, preds in forecast_results.items():
    forecast_df[f'{name}_Forecast'] = preds

print("\nForecast for the next 30 days:\n")
print(forecast_df.to_string(index=False))
