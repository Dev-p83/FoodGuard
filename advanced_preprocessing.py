"""
ADVANCED MILK QUALITY PREPROCESSING
Step-by-step improvement of your data
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🥛 ADVANCED MILK QUALITY PREPROCESSING")
print("="*60)

# ============================================
# STEP 1: Load and clean original data
# ============================================
print("\n📂 STEP 1: Loading original data...")
df = pd.read_csv("data/milknew.csv")
print(f"   Original: {df.shape[0]} rows, {df.shape[1]} columns")

# Print column names to see exact names
print(f"\n   Column names: {list(df.columns)}")

# Remove duplicates
print("\n🔍 STEP 2: Removing duplicates...")
before = len(df)
df = df.drop_duplicates()
print(f"   Removed {before - len(df)} duplicate rows")
print(f"   Now: {len(df)} unique samples")

# ============================================
# STEP 2: Handle outliers (fix invalid values)
# ============================================
print("\n📊 STEP 3: Handling outliers...")

# Store original count
original_count = len(df)

# Fix pH (normal milk pH is 5.5-8.5)
invalid_ph = df[(df['pH'] < 5.5) | (df['pH'] > 8.5)]
print(f"   pH outliers: {len(invalid_ph)}")
df = df[(df['pH'] >= 5.5) & (df['pH'] <= 8.5)]

# Fix Temperature (fresh milk should be 0-45°C)
invalid_temp = df[(df['Temprature'] < 0) | (df['Temprature'] > 45)]
print(f"   Temperature outliers: {len(invalid_temp)}")
df = df[(df['Temprature'] >= 0) & (df['Temprature'] <= 45)]

# Fix Colour (should be 200-255 for normal milk)
invalid_colour = df[(df['Colour'] < 200) | (df['Colour'] > 255)]
print(f"   Colour outliers: {len(invalid_colour)}")
df = df[(df['Colour'] >= 200) & (df['Colour'] <= 255)]

print(f"   Total rows after outlier removal: {len(df)}")
print(f"   Removed {original_count - len(df)} outlier rows")

# ============================================
# STEP 3: Create new features (Feature Engineering)
# ============================================
print("\n⚙️ STEP 4: Creating new features...")

# Note: Column name is 'Fat ' with a space!
# Let me check the exact column names
fat_col = 'Fat '  # Yes, there's a space

# Feature 1: pH to Temperature ratio
df['pH_Temp_Ratio'] = df['pH'] / df['Temprature']

# Feature 2: Combined quality score
df['Quality_Score'] = df['Taste'] + df['Odor'] + df[fat_col]

# Feature 3: Normalized colour (0-1 scale)
df['Colour_Normalized'] = df['Colour'] / 255

# Feature 4: Taste and Odor interaction
df['Taste_Odor_Interaction'] = df['Taste'] * df['Odor']

# Feature 5: Adulteration risk score (0-1)
df['Adulteration_Risk'] = (
    ((df['pH'] < 6.5) * 0.3) +        # Low pH = souring risk
    ((df['Turbidity'] == 1) * 0.3) +   # Cloudy = adulteration risk
    ((df['Taste'] == 0) * 0.4)         # Bad taste = spoilage risk
)

print("   Created 5 new features:")
print("      • pH_Temp_Ratio")
print("      • Quality_Score")
print("      • Colour_Normalized")
print("      • Taste_Odor_Interaction")
print("      • Adulteration_Risk")

# ============================================
# STEP 4: Prepare features for training
# ============================================
print("\n🎯 STEP 5: Preparing features...")

# Define target column
target_col = 'Grade'

# Select features (include new ones, exclude old ones if redundant)
feature_cols = ['pH', 'Temprature', 'Taste', 'Odor', fat_col, 'Turbidity', 
                'Colour', 'pH_Temp_Ratio', 'Quality_Score', 'Colour_Normalized',
                'Taste_Odor_Interaction', 'Adulteration_Risk']

X = df[feature_cols]
y = df[target_col]

print(f"   Using {len(feature_cols)} features for training")

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"   Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   Features scaled successfully")

# ============================================
# STEP 5: Train with Cross-Validation
# ============================================
print("\n🤖 STEP 6: Training with Cross-Validation...")

model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')

print(f"   Cross-validation scores: {cv_scores}")
print(f"   Average accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# Train final model on all data
model.fit(X_scaled, y_encoded)
print("   Final model trained on all data")

# ============================================
# STEP 6: Feature importance (which features matter most)
# ============================================
print("\n⭐ STEP 7: Feature Importance Analysis...")

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   Top 10 most important features:")
for i, row in importance_df.head(10).iterrows():
    print(f"      {row['Feature']}: {row['Importance']:.3f}")

# ============================================
# STEP 7: Save everything
# ============================================
print("\n💾 STEP 8: Saving models and preprocessing objects...")

joblib.dump(model, 'milk_model_advanced.pkl')
joblib.dump(scaler, 'scaler_advanced.pkl')
joblib.dump(le, 'label_encoder_advanced.pkl')
joblib.dump(feature_cols, 'feature_columns.pkl')

print("   ✓ Saved: milk_model_advanced.pkl")
print("   ✓ Saved: scaler_advanced.pkl")
print("   ✓ Saved: label_encoder_advanced.pkl")
print("   ✓ Saved: feature_columns.pkl")

# ============================================
# STEP 8: Summary report
# ============================================
print("\n" + "="*60)
print("📊 FINAL SUMMARY")
print("="*60)
print(f"""
Original data:    1059 rows (with 976 duplicates)
After cleaning:   {len(df)} unique, valid samples
Features used:    {len(feature_cols)} (5 new features created)
Model accuracy:   {cv_scores.mean():.2%} (cross-validated)

Key improvements made:
✅ Removed duplicates
✅ Fixed outlier values (pH, temperature, colour)
✅ Created 5 new informative features
✅ Used cross-validation for honest accuracy
✅ Saved all preprocessing objects
""")

print("✅ ADVANCED PREPROCESSING COMPLETE!")
print("="*60)