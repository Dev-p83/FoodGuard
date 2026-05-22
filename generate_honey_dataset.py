import pandas as pd
import numpy as np

np.random.seed(42)

rows = 4000

data = {
    "Moisture": np.random.uniform(10, 30, rows),
    "Sugar_Content": np.random.uniform(50, 95, rows),
    "pH": np.random.uniform(3, 6, rows),
    "Density": np.random.uniform(1.2, 1.6, rows),
    "Sucrose": np.random.uniform(0, 20, rows),
    "Ash_Content": np.random.uniform(0, 1, rows),
    "Electrical_Conductivity": np.random.uniform(0.1, 1.5, rows),
}

df = pd.DataFrame(data)

# AI LOGIC

df["Is_Adulterated"] = np.where(
    (df["Moisture"] > 22) |
    (df["Sucrose"] > 10) |
    (df["Sugar_Content"] < 60),
    1,
    0
)

df.to_csv("data/honey_dataset.csv", index=False)

print("Honey dataset created successfully!")
print(df.head())