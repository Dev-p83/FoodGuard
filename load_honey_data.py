# Step 1: Data Collection/Loading
import pandas as pd
import numpy as np

# Load the generated dataset (assuming it's saved as CSV)
# If not saved yet, let's save it first
from sklearn.datasets import make_classification

# Regenerate and save your dataset (since we need it saved)
np.random.seed(42)
n_samples = 1000

# Generate realistic honey features
moisture = np.random.uniform(13, 29, n_samples)
sugar_content = np.random.uniform(56, 88, n_samples)
pH = np.random.uniform(3.2, 6.8, n_samples)
density = np.random.uniform(1.23, 1.52, n_samples)
sucrose = np.random.uniform(0.01, 20, n_samples)
ash_content = np.random.uniform(0.18, 0.75, n_samples)
electrical_conductivity = np.random.uniform(0.15, 1.5, n_samples)

# Create adulterated samples (30% of data)
is_adulterated = np.zeros(n_samples)
adulterated_indices = np.random.choice(n_samples, int(n_samples * 0.3), replace=False)
is_adulterated[adulterated_indices] = 1

# Modify features for adulterated honey
moisture[adulterated_indices] = np.random.uniform(20, 35, len(adulterated_indices))
sugar_content[adulterated_indices] = np.random.uniform(75, 95, len(adulterated_indices))
pH[adulterated_indices] = np.random.uniform(4.5, 7.5, len(adulterated_indices))
density[adulterated_indices] = np.random.uniform(1.10, 1.35, len(adulterated_indices))

# Create DataFrame
df = pd.DataFrame({
    'Moisture': moisture,
    'Sugar_Content': sugar_content,
    'pH': pH,
    'Density': density,
    'Sucrose': sucrose,
    'Ash_Content': ash_content,
    'Electrical_Conductivity': electrical_conductivity,
    'Is_Adulterated': is_adulterated
})

# Save the dataset
df.to_csv('honey_dataset.csv', index=False)
print(f"Dataset saved! Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())