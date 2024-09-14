"""
This script processes and transforms a dataset from a CSV file to prepare it for machine learning tasks. It performs the following transformations:

1. **Remove Duplicates**: Removes duplicate rows from the dataset.
2. **Check and Handle Missing Data**: 
    Fills missing numerical values with the median and categorical values with the mode.
3. **Handle Special Characters**: 
    Removes non-numeric characters (e.g., currency symbols) from specified columns and converts the cleaned columns to float type.
4. **Handle Date Columns**: 
    Converts date columns to numeric representations and normalizes them between 0 and 1. Invalid dates are either removed or handled based on input.
5. **Handle Non-Numeric Data**: 
    Replaces unknown values (like 'N/A', 'unknown') with NaN, fills missing values, and applies label encoding for columns with a low number of unique values. One-hot encoding is applied to columns with a high number of unique values.
6. **Label Encoding for Target Column**: 
    Encodes the specified target column using label encoding.
7. **One-Hot Encoding**: 
    Applies one-hot encoding to remaining categorical columns.
8. **Normalization**: 
    Normalizes numeric columns using MinMaxScaler to scale them between 0 and 1.
9. **Log Transformation**: 
    Applies logarithmic transformation to numeric columns with only positive values.

### Functions:
- `remove_duplicates(data)`: Removes duplicate rows from the dataset.
- `check_missing_data(data)`: Fills missing data by column type (categorical: mode, numerical: median).
- `specialchar(data, specialchar_col)`: Cleans specified columns by removing non-numeric characters and converts them to float type.
- `handle_dates(data, date_column, drop_invalid_dates)`: Converts a date column to numeric representation and normalizes it.
- `handle_non_numeric_data(data, unknown_values)`: Replaces unknown values, fills missing categorical data, and applies appropriate encoding (label encoding or one-hot encoding).
- `label_encoding(data, target_column)`: Encodes the target column using label encoding.
- `one_hot_encoding(data)`: Applies one-hot encoding to remaining categorical columns.
- `normalization(data)`: Normalizes numeric data using MinMaxScaler.
- `standardization(data)`: Standardizes numeric data using StandardScaler.
- `log_transform(data)`: Applies log transformation to numeric columns with positive values.
- `transform_data(data, target_column, date_column)`: Main function that calls the above functions to perform a full transformation on the dataset.

### Usage:
1. The script prompts the user to enter the file path, target column, and date column if available.
2. The dataset is loaded from the specified CSV file, cleaned, transformed, and encoded.
3. The transformed data is saved to a new CSV file (`transformed_data.csv`) in the same directory as the input file.

"""

import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Input path for CSV file
path = input("Enter the path of the merged file CSV location: ")
data = pd.read_csv(path+'merged.csv')

# Replace this with the actual target column and date column if any
target_column = input("Enter the target column for label encoding: ")
date_column = " "
# Remove duplicate values
def remove_duplicates(data):
    ini_rows = data.shape[0]  # Number of rows before duplicate removal
    data = data.drop_duplicates()   # Remove duplicates
    final_rows = data.shape[0]   # Number of rows after duplicate removal
    print(f"Removed {ini_rows - final_rows} duplicate rows.")
    return data

# Check and handle missing data
def check_missing_data(data):
    missing_data = data.isnull().sum()  # Count missing values in each column
    print("Missing values in each column:\n", missing_data)
    
    # Example: Fill numerical columns with median, categorical with mode
    for col in data.columns:
        if data[col].dtype == 'object':  # For categorical columns
            mode_value = data[col].mode()[0]
            data[col].fillna(mode_value, inplace=True)
        else:  # For numerical columns
            median_value = data[col].median()
            data[col].fillna(median_value, inplace=True)
            
    return data

# Handle special characters in specific columns (e.g., 'Price')
def specialchar(data, specialchar_columns):
    for col in specialchar_columns:
        # Convert the column to string to use .str accessor
        data[col] = data[col].astype(str).str.replace(r'[^\d.]', '', regex=True)  # Remove non-numeric characters
    return data
# Handle date columns
def handle_dates(data, date_column, drop_invalid_dates=True):
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    
    if drop_invalid_dates:
        data.dropna(subset=[date_column], inplace=True)
    else:
        # Optionally handle invalid dates in another way
        pass
    
    min_date = data[date_column].min()
    max_date = data[date_column].max()
    range_days = (max_date - min_date).days
    
    # Convert dates to the number of days since the minimum date
    data[date_column] = (data[date_column] - min_date).dt.days
    
    # Normalize the date column between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[[date_column]] = scaler.fit_transform(data[[date_column]])
    
    return data

# Handle non-numeric data

def replace_null_values(data, unknown_values=['unknown', 'N/A', 'null', '?']):
    # Replace unknown values with NaN
    data.replace(unknown_values, np.nan, inplace=True)
    
    # Fill missing values based on column type
    for col in data.columns:
        if data[col].dtype == 'object':  # Categorical columns
            mode_value = data[col].mode()[0]
            data[col].fillna(mode_value, inplace=True)
            print(f"Filled missing values in column '{col}' with mode: {mode_value}")
        else:  # Numerical columns
            median_value = data[col].median()
            data[col].fillna(median_value, inplace=True)
            print(f"Filled missing values in column '{col}' with median: {median_value}")
    
    return data

# Label encode the target column
def label_encoding(data, target_column):
    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])
    print(f"Label encoded column '{target_column}'")
    return data

# Apply One-Hot Encoding to non-numeric columns
def one_hot_encoding(data, max_unique_values=100):
    # Select categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    # Filter out columns with more than 'max_unique_values' unique categories
    columns_to_encode = [col for col in categorical_columns if data[col].nunique() <= max_unique_values]
    columns_to_drop = [col for col in categorical_columns if data[col].nunique() > max_unique_values]
    
    print(f"Skipping one-hot encoding for columns with too many unique values: {columns_to_drop}")
    
    # OneHotEncoder for selected columns
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(data[columns_to_encode])
    
    # Create a dataframe for the one-hot encoded columns
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns_to_encode))
    
    # Reset index of the one-hot dataframe and original data to avoid alignment issues
    one_hot_df = one_hot_df.reset_index(drop=True)
    data = data.reset_index(drop=True)
    
    # Concatenate the one-hot encoded dataframe with the original dataframe
    df_encoded = pd.concat([data, one_hot_df], axis=1)
    
    # Drop the original categorical columns that were encoded
    df_encoded = df_encoded.drop(columns_to_encode + columns_to_drop, axis=1)
    
    print(f"Applied One-Hot Encoding to columns: {columns_to_encode}")
    return df_encoded

# Normalize numeric data
def normalization(data):
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    print("Normalized numeric data")
    return data

# Standardize numeric data
def standardization(data):
    scaler = preprocessing.StandardScaler()
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    print("Standardized numeric data")
    return data

# Apply log transformation to numeric columns
def log_transform(data):
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if (data[col] > 0).all():  # Only apply log transform to positive values
            data[f'log_{col}'] = np.log1p(data[col])  # log1p handles zero values
            print(f"Log transformation applied to '{col}'")
        else:
            print(f"Skipping log transformation for '{col}' due to zero or negative values.")
    return data

# Main function to transform the data
def transform_data(data, target_column=None, date_column=None, specialchar_columns=None):
    """
    This function applies a series of transformations to the input dataset to prepare it for machine learning tasks.
    
    Parameters:
    - data (pd.DataFrame): The dataset to transform.
    - target_column (str): The name of the target column for label encoding.
    - date_column (str): The name of the date column to process and normalize.
    - specialchar_columns (list of str): List of columns where special characters need to be removed.
    
    Returns:
    - pd.DataFrame: The transformed dataset.
    """
    
    print("Starting data transformation...")
    
    # Step 1: Remove duplicates
    data = remove_duplicates(data)
    
    # Step 2: Handle missing values
    data = check_missing_data(data)
    
    # Step 3: Handle special characters
    if specialchar_columns:
        data = specialchar(data, specialchar_columns)
    
    # Step 4: Handle date column (if provided)
    if date_column and date_column in data.columns:
        data = handle_dates(data, date_column)
    
    # Step 5: Handle non-numeric data (replace unknown values, fill missing, encode)
    data = replace_null_values(data)
    
    # Step 6: Apply One-Hot Encoding to remaining categorical columns
    data = one_hot_encoding(data)
    
    # Step 7: Label encode the target column (if provided)
    if target_column and target_column in data.columns:
        data = label_encoding(data, target_column)
    
    # Step 8: Normalize numeric data
    data = normalization(data)
    
    # Step 9: Apply log transformation to numeric columns with positive values
    data = log_transform(data)
    
    print("Data transformation complete.")
    return data
# Apply transformation
transformed_data = transform_data(data, target_column=target_column, date_column=date_column)

# Save the transformed data to a new CSV file
output_path = os.path.join('.', 'transformed_data.csv')
transformed_data.to_csv(output_path, index=False)
print(f"Transformed data saved to {output_path}")
