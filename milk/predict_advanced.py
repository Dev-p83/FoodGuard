"""
Make predictions using the advanced model
"""

import joblib
import pandas as pd
import numpy as np

# Load everything
model = joblib.load('milk_model_advanced.pkl')
scaler = joblib.load('scaler_advanced.pkl')
encoder = joblib.load('label_encoder_advanced.pkl')
feature_cols = joblib.load('feature_columns.pkl')

print("="*50)
print("🥛 ADVANCED MILK QUALITY PREDICTOR")
print("="*50)

# Get user input
print("\n📝 Enter milk sample measurements:")
print("-"*40)

ph = float(input("pH value (5.5-8.5): "))
temp = float(input("Temperature °C (0-45): "))
taste = int(input("Taste (0=Bad, 1=Good): "))
odor = int(input("Odor (0=Bad, 1=Good): "))
fat = int(input("Fat (0=Low, 1=High): "))
turbidity = int(input("Turbidity (0=Clear, 1=Cloudy): "))
colour = int(input("Colour value (200-255): "))

# Create DataFrame with original features (using exact column names)
input_df = pd.DataFrame([[ph, temp, taste, odor, fat, turbidity, colour]],
                        columns=['pH', 'Temprature', 'Taste', 'Odor', 'Fat ', 'Turbidity', 'Colour'])

print("\n🔧 Creating advanced features...")

# Create the same engineered features
input_df['pH_Temp_Ratio'] = input_df['pH'] / input_df['Temprature']
input_df['Quality_Score'] = input_df['Taste'] + input_df['Odor'] + input_df['Fat ']
input_df['Colour_Normalized'] = input_df['Colour'] / 255
input_df['Taste_Odor_Interaction'] = input_df['Taste'] * input_df['Odor']
input_df['Adulteration_Risk'] = (
    ((input_df['pH'] < 6.5) * 0.3) +
    ((input_df['Turbidity'] == 1) * 0.3) +
    ((input_df['Taste'] == 0) * 0.4)
)

print("✓ Features created successfully")

# Select only the features used in training
input_features = input_df[feature_cols]

# Scale the features
input_scaled = scaler.transform(input_features)

# Make prediction
prediction_num = model.predict(input_scaled)[0]
prediction_label = encoder.inverse_transform([prediction_num])[0]

# Get confidence score
probabilities = model.predict_proba(input_scaled)[0]
confidence = max(probabilities) * 100

print("\n" + "="*50)
print(f"📊 PREDICTION: {prediction_label.upper()} QUALITY")
print(f"🎯 Confidence: {confidence:.1f}%")
print("="*50)

# Show risk assessment
risk = input_df['Adulteration_Risk'].values[0]
if risk > 0.5:
    print("⚠️ HIGH adulteration risk detected!")
    print("   Recommended: Reject this batch")
elif risk > 0.3:
    print("⚠️ MEDIUM adulteration risk")
    print("   Recommended: Further testing needed")
else:
    print("✅ LOW adulteration risk")
    print("   Recommended: Accept this batch")

# Show detailed breakdown
print("\n📋 Detailed Analysis:")
print(f"   Quality Score: {input_df['Quality_Score'].values[0]}/3")
print(f"   Adulteration Risk: {risk:.1%}")
print(f"   pH/Temperature Ratio: {input_df['pH_Temp_Ratio'].values[0]:.2f}")