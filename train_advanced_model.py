import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# LOAD DATASET
df = pd.read_csv("data/final_milk_dataset.csv")

# FEATURES & TARGET
X = df.drop("Is_Adulterated", axis=1)
y = df["Is_Adulterated"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# FEATURE SCALING
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RANDOM FOREST MODEL
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)

# PREDICTION
rf_predictions = rf_model.predict(X_test_scaled)

# ACCURACY
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRANDOM FOREST ACCURACY:")
print(rf_accuracy)

# CLASSIFICATION REPORT
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, rf_predictions))

# CONFUSION MATRIX
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, rf_predictions))

# XGBOOST MODEL
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

# XGBOOST PREDICTIONS
xgb_predictions = xgb_model.predict(X_test_scaled)

# XGBOOST ACCURACY
xgb_accuracy = accuracy_score(y_test, xgb_predictions)

print("\nXGBOOST ACCURACY:")
print(xgb_accuracy)

# SAVE MODELS
joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(xgb_model, "models/xgboost_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nMODELS SAVED SUCCESSFULLY!")