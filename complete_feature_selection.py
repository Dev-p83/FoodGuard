# Step 6 Complete: Save Feature Selection Results
import pandas as pd
import numpy as np

print("="*60)
print("🍯 COMPLETING FEATURE SELECTION 🍯")
print("="*60)

# Load the engineered dataset
honey_df = pd.read_csv('honey_dataset_engineered.csv')

# Selected features from voting (10 best features)
selected_features = [
    'Moisture',
    'Sugar_Content', 
    'pH',
    'Density',
    'Moisture_Deficit',
    'pH_Sweetness',
    'Sugar_Moisture_Ratio',
    'Conductivity_per_Sugar',
    'Purity_Score',
    'Density_Index'
]

# Create final dataset with selected features
X_selected = honey_df[selected_features]
y = honey_df['Is_Adulterated']

# Save final dataset
final_df = pd.concat([X_selected, y], axis=1)
final_df.to_csv('honey_dataset_final.csv', index=False)

print("\n✅ Final dataset saved: 'honey_dataset_final.csv'")
print(f"   Shape: {final_df.shape}")
print(f"   Features: {len(selected_features)}")
print(f"   Target: Is_Adulterated")

print("\n📋 Selected Features (10 best):")
for i, feature in enumerate(selected_features, 1):
    print(f"   {i:2d}. {feature}")

# Show first few rows
print("\n📊 Preview of final dataset:")
print(final_df.head())

# Save feature list
with open('selected_features.txt', 'w') as f:
    f.write("Selected Features for Honey Adulteration Detection\n")
    f.write("="*50 + "\n")
    f.write("These 10 features were selected by consensus voting (5 methods):\n\n")
    for feature in selected_features:
        f.write(f"• {feature}\n")

print("\n💾 Feature list saved: 'selected_features.txt'")

print("\n" + "="*60)
print("✅ FEATURE SELECTION COMPLETE!")
print("="*60)
print("\n📊 Summary:")
print(f"   Original features: 15")
print(f"   Selected features: 10")
print(f"   Removed features: 5 (redundant or low importance)")
print("\n🎯 Ready for Model Training (Step 7)!")