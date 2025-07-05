# --- 1. Import All Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Machine Learning & Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- 2. Data Loading and Preprocessing ---
# Load dataset and set Date as a datetime index
try:
    # Replace with your file path
    df = pd.read_csv(r'C:\vscode\python-workspace\power_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
except FileNotFoundError:
    print("Dataset not found. A dummy dataset will be created for demonstration.")
    # Creating a dummy dataframe for demonstration purposes if file is not found
    dates = pd.date_range(start='2021-01-01', periods=730, freq='D')
    # Generate data with some seasonality and trend
    data = 50 + np.arange(730) * 0.05 + np.sin(np.arange(730) * 2 * np.pi / 365) * 10 + np.random.randn(730) * 5
    df = pd.DataFrame({'Power_Consumption': data}, index=dates)


# Resample to daily average consumption and remove missing values
df = df.resample('D').mean().dropna()

# --- 3. Exploratory Data Analysis (EDA) ---
# Plot the entire time series to observe overall trends and seasonality
print("Displaying overall time series data...")
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Power_Consumption'], label='Daily Power Consumption')
plt.title('Full Daily Power Consumption Time Series')
plt.xlabel('Date')
plt.ylabel('Power Consumption')
plt.grid(True)
plt.legend()
plt.show()

# --- 4. Feature Engineering & Data Splitting ---
# Create time-based features for the machine learning models
df['Day'] = df.index.dayofyear
df['Weekday'] = df.index.weekday
df['Month'] = df.index.month

# Define target (y) and features (X)
y = df['Power_Consumption']
X = df[['Day', 'Weekday', 'Month']]

# Split data into training and testing sets (80/20 split)
# We set shuffle=False to maintain the chronological order for time series data
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Scale features for distance-based algorithms like SVM and k-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Model Training and Evaluation ---
# Initialize models
ml_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVM': SVR(),
    'k-NN': KNeighborsRegressor(n_neighbors=7) # Using 7 neighbors as an example
}

# Dictionaries to store results and predictions
model_results = {}
model_predictions = {}

# Train, predict, and evaluate Machine Learning models
print("\nTraining Machine Learning models...")
for name, model in ml_models.items():
    # SVM and k-NN use scaled data
    if name in ['SVM', 'k-NN']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    model_predictions[name] = y_pred
    model_results[name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    }
print("Done.")

# Train, predict, and evaluate Time Series models
print("\nTraining Time Series models...")
# ARIMA Model (p,d,q) - using common starting parameters
try:
    arima_model = ARIMA(y_train, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=len(y_test))
    model_predictions['ARIMA'] = arima_forecast
    model_results["ARIMA"] = {
        'MAE': mean_absolute_error(y_test, arima_forecast),
        'RMSE': np.sqrt(mean_squared_error(y_test, arima_forecast)),
        'R²': r2_score(y_test, arima_forecast)
    }
except Exception as e:
    print(f"ARIMA model failed: {e}")

# SARIMA Model (p,d,q)(P,D,Q,m) - seasonal parameters (m=12 for monthly seasonality)
try:
    sarima_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.forecast(steps=len(y_test))
    model_predictions['SARIMA'] = sarima_forecast
    model_results["SARIMA"] = {
        'MAE': mean_absolute_error(y_test, sarima_forecast),
        'RMSE': np.sqrt(mean_squared_error(y_test, sarima_forecast)),
        'R²': r2_score(y_test, sarima_forecast)
    }
except Exception as e:
    print(f"SARIMA model failed: {e}")
print("Done.")

# --- 6. Results Visualization and Summary ---
# Plot all model predictions against the actual values on one graph
print("\nGenerating combined model forecast plot...")
plt.figure(figsize=(20, 10))
plt.plot(y_test.index, y_test.values, label='Actual Values', color='black', linewidth=2.5, zorder=5)

# Plot predictions for each model
for name, pred in model_predictions.items():
    plt.plot(y_test.index, pred, label=f'{name} Prediction', linestyle='--')

plt.title('Combined Model Forecast vs. Actual Power Consumption', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Power Consumption', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Display the performance metrics in a formatted table
print("\n--- Model Evaluation Summary ---")
results_df = pd.DataFrame(model_results).T
# Sort by R² score to easily see the best performing models
results_df.sort_values(by='R²', ascending=False, inplace=True)
print(results_df.round(4))