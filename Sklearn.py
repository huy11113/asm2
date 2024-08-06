import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample sales data
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'AveragePrice': [200, 220, 210, 230, 240, 250, 260, 270, 280, 290, 300, 310],
    'TotalUnitsSold': [500, 520, 510, 530, 540, 550, 560, 570, 580, 590, 600, 610]
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Select features and target
X = df[['Month', 'AveragePrice']]  # Features: Month and Average Price
y = df['TotalUnitsSold']  # Target: Total Units Sold

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot(y_test, y_test, color='red', linestyle='--')  # Diagonal line for perfect prediction
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.show()

# Predict future sales for upcoming months
future_months = pd.DataFrame({'Month': [13, 14, 15], 'AveragePrice': [320, 330, 340]})
future_sales_predictions = model.predict(future_months)

print("Predicted sales for upcoming months:", future_sales_predictions)
