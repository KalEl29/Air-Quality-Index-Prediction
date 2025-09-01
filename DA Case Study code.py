#Air Quality Index Prediction - Case Study

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load Dataset
df = pd.read_csv(r"C:\Users\netal\Documents\MITWPU\Assignments\DA\global_air_quality_dataset.csv")

# Rename columns for easy coding
df = df.rename(columns={
    'PM2.5 (µg/m³)': 'PM2.5',
    'PM10 (µg/m³)': 'PM10',
    'NO2 (ppb)': 'NO2',
    'SO2 (ppb)': 'SO2',
    'CO (ppm)': 'CO',
    'O3 (ppb)': 'O3',
    'Temperature (°C)': 'temp',
    'Humidity (%)': 'humidity',
    'Wind Speed (m/s)': 'wind_speed'
})

print("Dataset loaded successfully")
print(df.head())

# Step 2: Basic Cleaning
print("Shape before cleaning:", df.shape)
df = df.dropna()
print("Shape after cleaning:", df.shape)

# Step 3: Exploratory Data Analysis (EDA)
print(df.describe())

# Histogram for AQI
plt.figure(figsize=(8,5))
plt.hist(df['AQI'], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of AQI")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.show()

# Line plot for AQI (first 100 records)
plt.plot(df['AQI'][:100], color='red')
plt.title("AQI Trend (first 100 records)")
plt.xlabel("Index")
plt.ylabel("AQI")
plt.show()

# Step 4: Features & Target
X = df[['PM2.5','PM10','NO2','SO2','O3','CO','temp','humidity','wind_speed']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Models

# Linear Regression
lin_reg = LinearRegression().fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Step 6: Evaluation
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

evaluate(y_test, y_pred_lin, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

# Step 7: Results Visualization
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='green')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Random Forest: Actual vs Predicted AQI")
plt.show()