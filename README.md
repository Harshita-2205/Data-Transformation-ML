# Data-Transformation-ML
## Overview

This Python script processes and transforms a dataset from a CSV file to prepare it for machine learning tasks. It applies various data cleaning and transformation techniques to handle missing values, special characters, date columns, non-numeric data, and encodes categorical variables. Additionally, it normalizes and log-transforms numeric columns as required. The script saves the transformed dataset to a new CSV file, ready for further analysis or modeling.

## Features
### Remove Duplicates: 
Eliminates duplicate rows from the dataset.
### Handle Missing Data: 
Fills missing numerical values with the median and categorical values with the mode.
### Handle Special Characters:
Cleans non-numeric characters (like currency symbols) from specific columns and converts them to a numeric format.
### Process Date Columns: 
Converts date columns to numeric form and normalizes them between 0 and 1.
### Handle Non-Numeric Data: 
Deals with unknown values, encodes categorical data using label encoding or one-hot encoding based on the number of unique values.
### Target Column Encoding: 
Encodes the specified target column using label encoding.
### One-Hot Encoding: 
One-hot encodes remaining categorical columns.
### Normalization: 
Normalizes numeric columns using MinMaxScaler.
### Log Transformation: 
Applies log transformation to numeric columns containing positive values.

## Requirements
Python 3.x
Pandas
NumPy
scikit-learn

** To install the required packages, run: **

```pip install pandas numpy scikit-learn ```

## Usage
### Prepare the Input CSV File:
Ensure that your input dataset is in CSV format and located in the directory of your choice.
The script expects the file to be named merged.csv.
Running the Script:

### Run the script from the terminal or an IDE.
You will be prompted to enter:
The path to the directory containing the input CSV file.
The name of the target column (for label encoding).
The name of the date column (optional).

### Output:

The script will process the dataset and save the transformed data to a new CSV file, transformed_data.csv, in the same directory as the input file.
**Example Input**
```bash
Enter the path of the merged file CSV location: /path/to/dataset
Enter the target column for label encoding: TargetColumn
Enter the date column (if any, else leave blank): DateColumn
```
**Example Output**
```bash
Removed 10 duplicate rows.
Missing values in each column:
Column1: 5
Column2: 0
Handled special characters in 'Price' and converted to float.
Label encoded column 'TargetColumn'
One-hot encoded column 'CategoryColumn' and added 5 columns.
Normalized numeric data.
Log transformation applied to 'ColumnX'.
Data transformation complete.
Transformed data saved to /path/to/dataset/transformed_data.csv
```

## Functions
### remove_duplicates(data)
Purpose: Removes duplicate rows from the dataset.
Input: DataFrame (data).
Output: Cleaned DataFrame without duplicate rows.

### check_missing_data(data)
Purpose: Fills missing values by column type.
Categorical columns: Mode.
Numerical columns: Median.
Input: DataFrame (data).
Output: DataFrame with missing values handled.

### specialchar(data, specialchar_col)
Purpose: Cleans specified columns by removing non-numeric characters and converts them to float type.
Input: DataFrame (data), List of columns (specialchar_col).
Output: DataFrame with cleaned columns.

### handle_dates(data, date_column, drop_invalid_dates=True)
Purpose: Converts a date column to numeric representation and normalizes it.
Input: DataFrame (data), Column name (date_column), Boolean (drop_invalid_dates).
Output: DataFrame with normalized date column.

### replace_null_values(data, unknown_values=['unknown', 'N/A', 'null', '?'])
Purpose: Replaces unknown values and encodes categorical columns.
Input: DataFrame (data), List of unknown values (unknown_values).
Output: DataFrame with encoded columns.

### label_encoding(data, target_column)
Purpose: Encodes the target column using label encoding.
Input: DataFrame (data), Column name (target_column).
Output: DataFrame with encoded target column.

### one_hot_encoding(data)
Purpose: Applies one-hot encoding to remaining categorical columns.
Input: DataFrame (data).
Output: DataFrame with one-hot encoded columns.

### normalization(data)
Purpose: Normalizes numeric columns using MinMaxScaler.
Input: DataFrame (data).
Output: DataFrame with normalized numeric data.

### log_transform(data)
Purpose: Applies log transformation to numeric columns with positive values.
Input: DataFrame (data).
Output: DataFrame with log-transformed numeric columns.

### transform_data(data, target_column=None, date_column=None)
Purpose: Main function that orchestrates the entire transformation process.
Input: DataFrame (data), Target column (target_column), Date column (date_column).
Output: Transformed DataFrame.

## Output
The transformed dataset will be saved as transformed_data.csv in the same directory as the input CSV file. The transformed file is ready for machine learning tasks and analysis.

## License
This project is licensed under the MIT License.
