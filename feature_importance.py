import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/final_milk_dataset.csv")

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

# Features
X = df.drop("Is_Adulterated", axis=1)

# Feature importance
importance = model.feature_importances_

# Create dataframe
feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort values
feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

# Print top features
print(feature_df.head(10))

# Plot graph
plt.figure(figsize=(12,6))

plt.bar(
    feature_df["Feature"][:10],
    feature_df["Importance"][:10]
)

plt.xticks(rotation=45)

plt.title("Top 10 Important Features")

plt.tight_layout()

plt.show()