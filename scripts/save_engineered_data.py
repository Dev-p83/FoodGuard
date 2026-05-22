# Quick script to save the engineered data
import pandas as pd
import numpy as np

print("="*50)
print("Saving Engineered Honey Dataset")
print("="*50)

# Load cleaned data
honey_df = pd.read_csv('honey_dataset_cleaned.csv')
print(f"✓ Loaded cleaned data: {honey_df.shape}")

# Create all 8 engineered features
print("✓ Creating engineered features...")

honey_df['Sugar_Moisture_Ratio'] = honey_df['Sugar_Content'] / honey_df['Moisture']
honey_df['Density_Index'] = honey_df['Density'] * honey_df['Sugar_Content']
honey_df['pH_Sweetness'] = honey_df['pH'] * (honey_df['Sugar_Content'] / 100)
honey_df['Conductivity_per_Sugar'] = honey_df['Electrical_Conductivity'] / (honey_df['Sugar_Content'] + 1e-6)
honey_df['Ash_Moisture_Ratio'] = honey_df['Ash_Content'] / honey_df['Moisture']

# Purity Score
honey_df['Purity_Score'] = (
    (1 - (honey_df['Moisture'] - honey_df['Moisture'].min()) / (honey_df['Moisture'].max() - honey_df['Moisture'].min())) * 0.3 +
    (honey_df['Sugar_Content'] - honey_df['Sugar_Content'].min()) / (honey_df['Sugar_Content'].max() - honey_df['Sugar_Content'].min()) * 0.3 +
    (honey_df['Density'] - honey_df['Density'].min()) / (honey_df['Density'].max() - honey_df['Density'].min()) * 0.4
)

honey_df['Sucrose_Ratio'] = honey_df['Sucrose'] / (honey_df['Sugar_Content'] + 1e-6)
honey_df['Moisture_Deficit'] = np.abs(honey_df['Moisture'] - 18.0)

# Save the file
honey_df.to_csv('honey_dataset_engineered.csv', index=False)
print(f"✓ File saved: honey_dataset_engineered.csv")
print(f"✓ Final shape: {honey_df.shape}")
print(f"✓ Total columns: {len(honey_df.columns)}")
print(f"✓ Engineered features added: 8")

# Show first 2 rows to verify
print("\n📋 Preview (first 2 rows):")
print(honey_df[['Moisture', 'Sugar_Content', 'Sugar_Moisture_Ratio', 'Purity_Score', 'Is_Adulterated']].head(2))

print("\n✅ Done! Now run: python honey_feature_selection.py")