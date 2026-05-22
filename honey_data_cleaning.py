# Step 2: Honey Dataset - Data Cleaning
import pandas as pd
import numpy as np

# Load the honey dataset
honey_df = pd.read_csv('honey_dataset.csv')
print("="*60)
print("🍯 HONEY DATASET - DATA CLEANING 🍯")
print("="*60)

# 1. Check for missing values
print("\n📊 1. Missing Values Check:")
print(honey_df.isnull().sum())

# 2. Check for duplicates
print(f"\n🔄 2. Duplicate Honey Samples: {honey_df.duplicated().sum()}")

# 3. Check data types
print("\n📋 3. Honey Features Data Types:")
print(honey_df.dtypes)

# 4. Check for outliers in honey parameters
print("\n🔍 4. Outlier Detection in Honey Parameters (IQR method):")
honey_numeric_cols = honey_df.select_dtypes(include=[np.number]).columns
honey_outliers = {}

for col in honey_numeric_cols:
    Q1 = honey_df[col].quantile(0.25)
    Q3 = honey_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = honey_df[(honey_df[col] < lower_bound) | (honey_df[col] > upper_bound)]
    honey_outliers[col] = len(outliers)
    print(f"  🍯 {col}: {len(outliers)} suspicious values")

# 5. Statistical summary of honey samples
print("\n📈 5. Statistical Summary of Honey Dataset:")
print(honey_df.describe())

# 6. Check for invalid values in honey measurements
print("\n✅ 6. Invalid Values Check (Negative measurements):")
for col in honey_numeric_cols:
    if col != 'Is_Adulterated':
        negative_count = (honey_df[col] < 0).sum()
        if negative_count > 0:
            print(f"  ⚠️ {col}: {negative_count} negative values (physically impossible!)")
        else:
            print(f"  ✓ {col}: All values valid")

# 7. Handle outliers (capping method for honey parameters)
print("\n🔧 7. Handling Outliers in Honey Data (Capping at 99th percentile)...")
honey_cleaned = honey_df.copy()

for col in honey_numeric_cols:
    if col != 'Is_Adulterated':
        lower_cap = honey_df[col].quantile(0.01)
        upper_cap = honey_df[col].quantile(0.99)
        honey_cleaned[col] = honey_cleaned[col].clip(lower_cap, upper_cap)
        print(f"  🍯 {col}: Capped range [{lower_cap:.2f}, {upper_cap:.2f}]")

# 8. Save cleaned honey dataset
honey_cleaned.to_csv('honey_dataset_cleaned.csv', index=False)
print("\n💾 ✅ Cleaned honey dataset saved as 'honey_dataset_cleaned.csv'")

# 9. Compare before and after cleaning
print("\n📊 8. Honey Dataset - Before vs After Cleaning:")
print(f"  Original honey samples: {honey_df.shape[0]}")
print(f"  Cleaned honey samples: {honey_cleaned.shape[0]}")
print(f"  Features preserved: {honey_df.shape[1]}")

print("\n🍯 First 5 rows of CLEANED honey data:")
print(honey_cleaned.head())

print("\n" + "="*60)
print("✅ HONEY DATA CLEANING COMPLETE! Ready for preprocessing.")
print("="*60)