import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


# Sample data (monthly sales)
# Assuming months 1 to 12 and corresponding sales figures
months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
sales = np.array([150, 180, 220, 260, 300, 320, 340, 400, 450, 500, 550, 600])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(months, sales, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales
sales_pred_train = model.predict(X_train)
sales_pred_test = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, sales_pred_test)
r2 = r2_score(y_test, sales_pred_test)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(months, sales, color='blue', label='Actual Sales')
plt.plot(X_train, sales_pred_train, color='green', label='Train Prediction')
plt.scatter(X_test, sales_pred_test, color='red', label='Test Prediction')
plt.title('Sales Prediction using Linear Regression')
plt.xlabel('Months')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()



# Sample data (monthly sales)
months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
sales = np.array([150, 180, 220, 260, 300, 320, 340, 400, 450, 500, 550, 600])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(months, sales, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future sales for the next 6 months
future_months = np.array([13, 14, 15, 16, 17, 18]).reshape(-1, 1)
future_sales_predictions = model.predict(future_months)

# Visualize the historical and predicted sales
plt.figure(figsize=(10, 6))
plt.scatter(months, sales, color='blue', label='Actual Sales')
plt.plot(months, model.predict(months), color='green', label='Fitted Line')
plt.scatter(future_months, future_sales_predictions, color='red', label='Future Predictions', marker='x')
plt.title('Sales Prediction using Linear Regression')
plt.xlabel('Months')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Print future sales predictions
for month, prediction in zip(future_months.flatten(), future_sales_predictions):
    print(f'Predicted sales for month {month}: {prediction:.2f}')
