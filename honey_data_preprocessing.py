# Step 3: Honey Dataset - Data Preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
import joblib

print("="*60)
print("🍯 HONEY DATASET - DATA PREPROCESSING 🍯")
print("="*60)

# Load the cleaned honey dataset
honey_df = pd.read_csv('honey_dataset_cleaned.csv')
print("\n📊 Loaded cleaned honey dataset:")
print(f"  Shape: {honey_df.shape}")
print(f"  Columns: {list(honey_df.columns)}")

# 1. Separate features (X) and target (y)
print("\n" + "="*60)
print("1️⃣  SEPARATING FEATURES AND TARGET")
print("="*60)

X = honey_df.drop('Is_Adulterated', axis=1)  # Honey features
y = honey_df['Is_Adulterated']  # Target (1=Adulterated, 0=Pure)

print(f"  Features (X) shape: {X.shape}")
print(f"  Features names: {list(X.columns)}")
print(f"  Target (y) shape: {y.shape}")
print(f"  Target distribution:")
print(f"    Pure honey (0): {(y==0).sum()} samples ({(y==0).mean()*100:.1f}%)")
print(f"    Adulterated (1): {(y==1).sum()} samples ({(y==1).mean()*100:.1f}%)")

# 2. Train-Test Split (70% train, 30% test)
print("\n" + "="*60)
print("2️⃣  TRAIN-TEST SPLIT (70-30)")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y  # Maintain same proportion of adulterated honey
)

print(f"  Training set - Features: {X_train.shape}, Target: {y_train.shape}")
print(f"  Testing set - Features: {X_test.shape}, Target: {y_test.shape}")
print(f"\n  Training set target distribution:")
print(f"    Pure: {(y_train==0).sum()} ({(y_train==0).mean()*100:.1f}%)")
print(f"    Adulterated: {(y_train==1).sum()} ({(y_train==1).mean()*100:.1f}%)")
print(f"\n  Testing set target distribution:")
print(f"    Pure: {(y_test==0).sum()} ({(y_test==0).mean()*100:.1f}%)")
print(f"    Adulterated: {(y_test==1).sum()} ({(y_test==1).mean()*100:.1f}%)")

# 3. Feature Scaling (3 different methods for comparison)
print("\n" + "="*60)
print("3️⃣  FEATURE SCALING")
print("="*60)

# Method 1: StandardScaler (Z-score normalization)
standard_scaler = StandardScaler()
X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)

# Method 2: MinMaxScaler (Normalize to [0,1] range)
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

# Method 3: RobustScaler (Robust to outliers)
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)

print("  ✅ Applied 3 scaling methods:")
print("    1. StandardScaler (mean=0, std=1)")
print("    2. MinMaxScaler (range [0,1])")
print("    3. RobustScaler (based on percentiles)")

# Show scaling example
print("\n  📊 Scaling example (first 3 samples, first feature 'Moisture'):")
print(f"    Original Moisture values: {X_train['Moisture'].iloc[:3].values}")
print(f"    StandardScaled: {X_train_standard[:3, 0]}")
print(f"    MinMaxScaled: {X_train_minmax[:3, 0]}")
print(f"    RobustScaled: {X_train_robust[:3, 0]}")

# 4. Save preprocessed datasets
print("\n" + "="*60)
print("4️⃣  SAVING PREPROCESSED DATASETS")
print("="*60)

# Save StandardScaler version (recommended for most ML algorithms)
np.save('X_train_standard.npy', X_train_standard)
np.save('X_test_standard.npy', X_test_standard)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Save dataframes for easy viewing
pd.DataFrame(X_train_standard, columns=X.columns).to_csv('X_train_standard.csv', index=False)
pd.DataFrame(X_test_standard, columns=X.columns).to_csv('X_test_standard.csv', index=False)

# Save scalers for future use (important for deployment)
joblib.dump(standard_scaler, 'honey_standard_scaler.pkl')
joblib.dump(minmax_scaler, 'honey_minmax_scaler.pkl')
joblib.dump(robust_scaler, 'honey_robust_scaler.pkl')

print("  ✅ Saved preprocessed datasets:")
print("    - X_train_standard.npy / .csv (training features)")
print("    - X_test_standard.npy / .csv (testing features)")
print("    - y_train.npy (training labels)")
print("    - y_test.npy (testing labels)")
print("    - honey_standard_scaler.pkl (scaler for deployment)")
print("    - honey_minmax_scaler.pkl")
print("    - honey_robust_scaler.pkl")

# 5. Verify preprocessing
print("\n" + "="*60)
print("5️⃣  VERIFICATION")
print("="*60)

print(f"  Original data range - Moisture: [{X['Moisture'].min():.2f}, {X['Moisture'].max():.2f}]")
print(f"  Scaled data range - Moisture: [{X_train_standard[:,0].min():.2f}, {X_train_standard[:,0].max():.2f}]")
print(f"  Scaled data mean - Moisture: {X_train_standard[:,0].mean():.2f}")
print(f"  Scaled data std - Moisture: {X_train_standard[:,0].std():.2f}")

print("\n" + "="*60)
print("✅ HONEY DATA PREPROCESSING COMPLETE!")
print("="*60)
print("\n📁 Generated files:")
print("  • Preprocessed numpy arrays (for ML models)")
print("  • CSV files (for inspection)")
print("  • Pickle files (scalers for deployment)")
print("\n🍯 Ready for Exploratory Data Analysis (EDA)!")