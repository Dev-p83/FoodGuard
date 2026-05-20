import pandas as pd

# Load large milk dataset
df = pd.read_csv("data/archive/milk_combined_full_dataset.csv")

# FIRST 5 ROWS
print("\nFIRST 5 ROWS:")
print(df.head())

# DATASET SHAPE
print("\nDATASET SHAPE:")
print(df.shape)

# COLUMN NAMES
print("\nCOLUMN NAMES:")
print(df.columns)

# DATASET INFO
print("\nDATASET INFO:")
print(df.info())

# MISSING VALUES
print("\nMISSING VALUES:")
print(df.isnull().sum())

# DUPLICATE ROWS
print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# STATISTICS
print("\nSTATISTICS:")
print(df.describe())