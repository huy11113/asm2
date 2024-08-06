import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


# Example of loading data from a CSV file
df = pd.read_csv('Customer Table.csv')

# View the first 5 rows of the data
print(df.head())

# Information about data types and missing values
print(df.info())

# Descriptive statistics
print(df.describe())

# Check for missing values in each column
print(df.isnull().sum())

# Drop rows with any missing values
df_cleaned = df.dropna()

# Alternatively, fill missing values with a default value (e.g., 'Unknown' for strings)
df_filled = df.fillna('Unknown')

# Example: Remove rows where the Email column does not contain '@'
df_cleaned = df[df['Email'].str.contains('@')]

# Example: Rename columns if needed for clarity
df.rename(columns={'FirstName': 'First_Name', 'LastName': 'Last_Name'}, inplace=True)

# Convert PostalCode and Phone to string type for consistency
df['PostalCode'] = df['PostalCode'].astype(str)
df['Phone'] = df['Phone'].astype(str)

df.to_csv('customer_data_cleaned.csv', index=False)

