import matplotlib.pyplot as plt
import pandas as pd

# Load the transformed customer data
df = pd.read_csv('customer_data_cleaned.csv')

# Sample data (ensure all lists have the same length)
data = {
    'ProductID': [1, 2, 3, 4, 5, 6, 7],
    'ProductName': ['Smartphone', 'Laptop', 'Tablet', 'Smartwatch', 'Headphones', 'Bluetooth Speaker', 'Television'],
    'Category': ['Electronics'] * 7,
    'UnitPrice': [500, 1000, 300, 200, 100, 150, 800],
    'InStock': [1000, 800, 1200, 1500, 2000, 1800, 600],
    'SupplierID': [1] * 7,
    'DateAdded': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create a scatter plot for UnitPrice vs InStock
plt.figure(figsize=(10, 6))
plt.scatter(df['UnitPrice'], df['InStock'], color='blue', alpha=0.7, s=100)

# Annotate each point with the ProductName
for i in range(len(df)):
    plt.annotate(df['ProductName'][i], (df['UnitPrice'][i], df['InStock'][i]), textcoords="offset points", xytext=(0,5), ha='center')

plt.title('Scatter Plot of Unit Price vs Stock Quantity')
plt.xlabel('Unit Price')
plt.ylabel('Stock Quantity')
plt.grid(True)
plt.tight_layout()
plt.show()

# Sample data (ensure all lists have the same length)
data = {
    'ProductID': [1, 2, 3, 4, 5, 6, 7],
    'ProductName': ['Smartphone', 'Laptop', 'Tablet', 'Smartwatch', 'Headphones', 'Bluetooth Speaker', 'Television'],
    'Category': ['Electronics'] * 7,
    'UnitPrice': [500, 1000, 300, 200, 100, 150, 800],
    'InStock': [1000, 800, 1200, 1500, 2000, 1800, 600],
    'SupplierID': [1] * 7,
    'DateAdded': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create a scatter plot for UnitPrice vs InStock
plt.figure(figsize=(10, 6))
plt.scatter(df['UnitPrice'], df['InStock'], color='blue', alpha=0.7, s=100)

# Annotate each point with the ProductName
for i in range(len(df)):
    plt.annotate(df['ProductName'][i], (df['UnitPrice'][i], df['InStock'][i]), textcoords="offset points", xytext=(0,5), ha='center')

plt.title('Scatter Plot of Unit Price vs Stock Quantity')
plt.xlabel('Unit Price')
plt.ylabel('Stock Quantity')
plt.grid(True)
plt.tight_layout()
plt.show()