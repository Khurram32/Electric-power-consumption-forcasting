Electric Power Consumption Forecasting
This project focuses on forecasting electric power consumption using various machine learning and time-series models. It evaluates the performance of different algorithms on a large dataset and provides an analysis of their effectiveness. The project also includes a Flask API to serve predictions from the best-performing model.

Table of Contents
Project Overview

Features

Models Implemented

Dataset

Key Findings

Getting Started

Prerequisites

Installation

Usage

Running the Main Script

Using the API

Project Structure

Contributing

Project Overview
The main objective of this project is to predict future electricity consumption based on historical data. This is achieved by implementing and comparing several popular forecasting models. The analysis highlights the strengths and weaknesses of each model in capturing the complex patterns of power usage. An ensemble model, Random Forest, was identified as the most effective and is deployed via a simple REST API.

Features
Data Preprocessing: Cleans and prepares the time-series data for modeling.

Feature Engineering: Creates time-based features like day of the year, weekday, month, and lag features to improve model accuracy.

Multi-Model Evaluation: Implements and evaluates 7 different forecasting models.

Performance Metrics: Uses MAE, RMSE, and R² to compare model performance.

Visualization: Generates plots to compare the predictions of each model against the actual data.

API for Predictions: A Flask-based API to get real-time predictions.

Models Implemented
The following models have been implemented and compared:

Machine Learning Models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Support Vector Machine (SVR)

k-Nearest Neighbors (k-NN)

Time-Series Models:

ARIMA (Autoregressive Integrated Moving Average)

SARIMA (Seasonal ARIMA)

Dataset
The project uses a dataset of electric power consumption with over 52,000 records. The data is time-series-based and contains power consumption values recorded over a period of time.

Source: Kaggle (or specify the exact source if known)

File: power_data.csv (or your_data.csv)

Key Findings
Random Forest and Decision Tree models were the most effective, closely tracking the actual consumption and capturing dynamic fluctuations.

k-NN showed decent performance but with a slight delay in reacting to rapid changes.

Linear Regression and SVM were less effective, producing flatter forecasts that didn't capture the data's volatility.

The analysis suggests that regular retraining of models with new data is crucial for maintaining prediction accuracy over time.

Ensemble methods like Random Forest are highly suitable for real-world load forecasting due to their robustness.

Getting Started
Follow these instructions to get a copy of the project up and running on your local machine.

Prerequisites
Make sure you have Python 3.6+ installed. You will also need pip to install the required libraries.

Installation
Clone the repository:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Install the required libraries:
Create a requirements.txt file with the following content:

pandas
numpy
scikit-learn
matplotlib
statsmodels
joblib
flask

Then, install the dependencies:

pip install -r requirements.txt

Usage
Running the Main Script
The completemodel.py or power_forecast_full.py script will perform the data loading, training, evaluation, and will also start the Flask API server.

Make sure your dataset (power_data.csv or your_data.csv) is in the correct directory as specified in the script.

Run the script from your terminal:

python completemodel.py

This will:

Train all the models.

Print the evaluation metrics for each model.

Generate and save plots for model comparisons.

Save the best model (best_rf_model.pkl) and the scaler (scaler.pkl).

Start the Flask API server.

Using the API
Once the server is running, you can interact with the following endpoints:

GET /: A simple endpoint to check if the API is running.

POST /predict: Get a single power consumption prediction.

URL: http://127.0.0.1:5000/predict

Method: POST

Body (JSON):

{
    "Day": 150,
    "Weekday": 3,
    "Month": 5,
    "Lag_1": 45.6,
    "Lag_7": 48.2,
    "Rolling_Mean_7": 46.5
}

Success Response:

{
    "Power_Consumption_Prediction": 50.25
}

GET /forecast: Get a 7-day forecast.

URL: http://127.0.0.1:5000/forecast

Method: GET

Success Response:

{
    "2023-01-01": 48.5,
    "2023-01-02": 49.1,
    ...
}

Project Structure
.
├── completemodel.py            # Main script for training, evaluation, and visualization
├── power_forecast_full.py      # Script with training and Flask API
├── powerdata_arime_model.py    # Script focusing on ARIMA/SARIMA models
├── best_rf_model.pkl           # Saved best performing model
├── scaler.pkl                  # Saved scaler object
├── model_metrics.csv           # CSV file with performance metrics of all models
├── *.png                       # Saved plots of model forecasts
└── requirements.txt            # Project dependencies

Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions for improvements.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request
