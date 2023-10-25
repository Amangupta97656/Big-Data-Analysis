import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the dataset (adjust the path to your dataset)
data = pd.read_csv('rainfall_india_1901-2015.csv')

# Prepare the dataset (feature selection and preprocessing)
X = data[['Year', 'Month']]  # Adjust features
y = data['Rainfall']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Regressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model (you can use different metrics)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Year'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Year'], y_pred, color='red', label='Predicted')
plt.title('Actual vs. Predicted Rainfall')
plt.xlabel('Year')
plt.ylabel('Rainfall')
plt.legend()
plt.show()


# Explore other advanced analysis techniques based on your project goals and dataset characteristics. Examples include:

# 1. Principal Component Analysis (PCA) for Dimensionality Reduction
from sklearn.decomposition import PCA

# Fit PCA to your data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the reduced-dimensional data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA: Dimensionality Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 2. Time Series Analysis
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Assuming you have a time series dataset, use ARIMA model as an example
model = sm.tsa.ARIMA(y, order=(1, 1, 1))
results = model.fit()

# Visualize ACF and PACF plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(results.resid, ax=ax1)
plot_pacf(results.resid, ax=ax2)
plt.show()

# 3. Advanced Visualizations
import seaborn as sns

# Example of a pairplot to visualize relationships between features
sns.pairplot(data, vars=['feature1', 'feature2'], hue='target_category')
plt.title('Pairplot of Features')
plt.show()


