import pandas as pd

# Read data from Excel file
file_path = 'path_to_your_excel_file.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataframe
print("First few rows of the dataframe:")
print(data.head())

# Summary statistics for numerical columns
print("\nSummary statistics:")
print(data.describe())

# Checking for missing values in each column
print("\nMissing values per column:")
print(data.isnull().sum())

# Example analysis: Group by a categorical column and calculate mean of a numeric column
# Assuming there is a categorical column named 'Category' and a numerical column named 'Value'
if 'Category' in data.columns and 'Value' in data.columns:
    print("\nMean value per category:")
    print(data.groupby('Category')['Value'].mean())
else:
    print("\nNo 'Category' or 'Value' columns found for analysis.")
