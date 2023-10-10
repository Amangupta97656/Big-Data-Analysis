# Python Code for Phase 2

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('rainfall in india 1901-2015.csv')

# Data Preparation
# Perform data preprocessing (e.g., handling missing values, feature engineering)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Algorithm Selection
# Initialize models
random_forest_model = RandomForestRegressor()
n_steps = 10  # Define the number of time steps for LSTM
n_features = 1  # Define the number of features for LSTM
lstm_model = keras.Sequential([
    layers.LSTM(units=64, activation='relu', input_shape=(n_steps, n_features)),
    layers.Dense(1)
])

# Training and Validation
# Train and validate the models

# Evaluate model performance (e.g., Mean Absolute Error, Root Mean Squared Error)


# Predictive Modeling (Example with Linear Regression)
# Generate synthetic data (Replace this with loading and preprocessing your actual data)
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the actual vs. predicted values
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()



# Python Code for Phase 2 (Continued)

# Anomaly Detection using Residuals
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset (use your dataset)
data = pd.read_csv('rainfall_india_1901-2015.csv')

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Feature selection and preprocessing (adjust features as needed)
X_train = train_data[['Year', 'Month', 'Region']]
y_train = train_data['Rainfall']
X_test = test_data[['Year', 'Month', 'Region']]
y_test = test_data['Rainfall']

# Initialize and train a machine learning model (Random Forest, for example)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate residuals (difference between actual and predicted values)
residuals = y_test - y_pred

# Define a threshold for anomaly detection (adjust as needed)
threshold = 2.0  # Example threshold, you can fine-tune this value

# Identify anomalies based on residuals
anomalies = np.where(np.abs(residuals) > threshold)[0]

# Print the indices of anomalous data points
print("Anomalous Data Points:")
print(anomalies)

# SQL Commands for Database Connection (assuming you've configured the connection)
# These are the SQL commands to connect to an IBM Db2 database using the VS Code extension:
# 1. Open a new SQL file in VS Code.
# 2. Use the following SQL command to connect to the database using your configured connection:
"""
CONNECT TO your_database_name
USER your_username
USING your_password
HOSTNAME your_hostname
PORT your_port
DATABASE your_database_name
"""

