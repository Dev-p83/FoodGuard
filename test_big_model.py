import joblib
import pandas as pd

model = joblib.load("models/big_foodgurd_model.pkl")
model_columns = joblib.load("models/big_model_columns.pkl")

df = pd.read_csv("data/archive/milk_combined_full_dataset.csv")

sample = df.iloc[[0]].copy()

drop_columns = [
    "Sample_ID",
    "Adulterant_Detected",
    "Is_Adulterated",
    "Quality_Score_0_100",
    "Adulteration_Index_0_1",
    "FSSAI_Chemical_Score",
]

meta_columns = [col for col in df.columns if "__META_" in col]
drop_columns = drop_columns + meta_columns

sample = sample.drop(columns=drop_columns, errors="ignore")

sample = pd.get_dummies(sample)
sample = sample.reindex(columns=model_columns, fill_value=0)

prediction = model.predict(sample)[0]

print("Prediction:", prediction)

if prediction == 1:
    print("Result: Adulterated")
else:
    print("Result: Safe")
