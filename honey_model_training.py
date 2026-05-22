# Step 7: Honey Dataset - Model Training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🍯 HONEY DATASET - MODEL TRAINING 🍯")
print("="*60)

# Load the preprocessed data (from Step 3)
X_train = np.load('X_train_standard.npy')
X_test = np.load('X_test_standard.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Load selected feature names (from Step 6)
try:
    with open('selected_features.txt', 'r') as f:
        feature_names = [line.strip().replace('• ', '') for line in f if '•' in line]
except:
    feature_names = ['Moisture', 'Sugar_Content', 'pH', 'Density', 'Moisture_Deficit', 
                     'pH_Sweetness', 'Sugar_Moisture_Ratio', 'Conductivity_per_Sugar', 
                     'Purity_Score', 'Density_Index']

print("\n📊 Data loaded successfully:")
print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"  Training labels: {y_train.shape[0]} samples")
print(f"  Testing labels: {y_test.shape[0]} samples")

print(f"\n📋 Features being used ({len(feature_names)} features):")
for i, feature in enumerate(feature_names[:10], 1):
    print(f"  {i:2d}. {feature}")

# ============================================================================
# 1. DEFINE MODELS
# ============================================================================
print("\n" + "="*60)
print("1️⃣  DEFINING MODELS")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier()
}

print("\n✅ Initialized 7 models:")
for model_name in models.keys():
    print(f"  • {model_name}")

# ============================================================================
# 2. TRAIN AND EVALUATE MODELS
# ============================================================================
print("\n" + "="*60)
print("2️⃣  TRAINING AND EVALUATING MODELS")
print("="*60)

results = []
cv_results = {}

for model_name, model in models.items():
    print(f"\n📊 Training {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # ROC-AUC (if probability available)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Store results
    result = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc if roc_auc is not None else 0,
        'CV Accuracy': cv_mean,
        'CV Std': cv_std
    }
    results.append(result)
    
    # Store model for later
    cv_results[model_name] = cv_scores
    
    print(f"  ✓ Accuracy: {accuracy:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")
    print(f"  ✓ CV Score: {cv_mean:.4f} (±{cv_std:.4f})")

# ============================================================================
# 3. RESULTS COMPARISON
# ============================================================================
print("\n" + "="*60)
print("3️⃣  RESULTS COMPARISON")
print("="*60)

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n📊 Model Performance Ranking (by Accuracy):")
print("="*80)
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
print("="*80)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<20} {row['Accuracy']:.4f}     {row['Precision']:.4f}     {row['Recall']:.4f}     {row['F1-Score']:.4f}     {row['ROC-AUC']:.4f}")

# ============================================================================
# 4. FIND BEST MODEL
# ============================================================================
print("\n" + "="*60)
print("4️⃣  BEST MODEL SELECTION")
print("="*60)

best_model_name = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']
best_f1 = results_df.iloc[0]['F1-Score']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f}")
print(f"   F1-Score: {best_f1:.4f}")

# Save best model
best_model = models[best_model_name]

# ============================================================================
# 5. CONFUSION MATRIX FOR BEST MODEL
# ============================================================================
print("\n" + "="*60)
print("5️⃣  CONFUSION MATRIX ANALYSIS")
print("="*60)

# Get predictions from best model
y_pred_best = best_model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_best)

print(f"\n📊 Confusion Matrix for {best_model_name}:")
print("="*40)
print(f"                 Predicted")
print(f"                 Pure    Adulterated")
print(f"Actual Pure     {cm[0,0]:4d}      {cm[0,1]:4d}")
print(f"       Adulterated {cm[1,0]:4d}      {cm[1,1]:4d}")

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # True Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate

print(f"\n📈 Detailed Metrics:")
print(f"  True Negatives (Pure correctly identified): {tn}")
print(f"  False Positives (Pure marked as adulterated): {fp}")
print(f"  False Negatives (Adulterated marked as pure): {fn}")
print(f"  True Positives (Adulterated correctly identified): {tp}")
print(f"  Sensitivity (Recall): {sensitivity:.4f}")
print(f"  Specificity: {specificity:.4f}")

# ============================================================================
# 6. HYPERPARAMETER TUNING FOR BEST MODEL
# ============================================================================
print("\n" + "="*60)
print("6️⃣  HYPERPARAMETER TUNING")
print("="*60)

print(f"\n🔧 Tuning {best_model_name} for better performance...")

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                               param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_tuned_model = grid_search.best_estimator_
    print(f"  Best parameters: {grid_search.best_params_}")
    
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    grid_search = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), 
                               param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_tuned_model = grid_search.best_estimator_
    print(f"  Best parameters: {grid_search.best_params_}")
    
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), 
                               param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_tuned_model = grid_search.best_estimator_
    print(f"  Best parameters: {grid_search.best_params_}")
    
else:
    best_tuned_model = best_model
    print("  No tuning performed for this model type")

# Evaluate tuned model
y_pred_tuned = best_tuned_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print(f"\n📊 Tuning Results:")
print(f"  Before tuning: {best_accuracy:.4f}")
print(f"  After tuning: {tuned_accuracy:.4f}")
print(f"  Improvement: {tuned_accuracy - best_accuracy:.4f}")

# ============================================================================
# 7. SAVE BEST MODEL
# ============================================================================
print("\n" + "="*60)
print("7️⃣  SAVING BEST MODEL")
print("="*60)

import joblib

# Save the best tuned model
joblib.dump(best_tuned_model, 'honey_best_model.pkl')
print("💾 Model saved: 'honey_best_model.pkl'")

# Also save the scaler (from Step 3)
import joblib
try:
    scaler = joblib.load('honey_standard_scaler.pkl')
    print("💾 Scaler loaded: 'honey_standard_scaler.pkl'")
except:
    print("⚠️ Scaler file not found, will need to retrain for deployment")

# Save model metadata
model_metadata = {
    'model_name': best_model_name,
    'accuracy': tuned_accuracy,
    'features': feature_names,
    'tuned': best_model_name != best_model_name if 'tuned_accuracy' else False
}

import json
with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)
print("💾 Model metadata saved: 'model_metadata.json'")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n" + "="*60)
print("8️⃣  CREATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Model Comparison (Top 5)
ax1 = axes[0, 0]
top5 = results_df.head(5)
colors = ['gold', 'silver', '#cd7f32', '#3498db', '#2ecc71']
bars = ax1.bar(top5['Model'], top5['Accuracy'], color=colors[:len(top5)])
ax1.set_ylim([0, 1])
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Performance Comparison', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
for bar, acc in zip(bars, top5['Accuracy']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', fontweight='bold')

# Plot 2: Confusion Matrix for Best Model
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax2, 
            xticklabels=['Pure', 'Adulterated'], 
            yticklabels=['Pure', 'Adulterated'])
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title(f'{best_model_name} - Confusion Matrix', fontweight='bold')

# Plot 3: Cross-validation Scores
ax3 = axes[1, 0]
cv_data = []
model_names = []
for model_name, scores in cv_results.items():
    cv_data.append(scores)
    model_names.append(model_name)

bp = ax3.boxplot(cv_data, labels=model_names, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax3.set_ylabel('Cross-Validation Accuracy')
ax3.set_title('5-Fold Cross-Validation Performance', fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.set_ylim([0.5, 1.0])

# Plot 4: ROC Curves (for top 3 models)
ax4 = axes[1, 1]
from sklearn.metrics import roc_curve

for model_name in results_df.head(3)['Model']:
    model = models[model_name]
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        ax4.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)

ax4.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curves - Top 3 Models', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle(f'Honey Adulteration Detection - Model Training Results', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('honey_eda_plots/09_model_training_results.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. SUMMARY REPORT
# ============================================================================
print("\n" + "="*60)
print("9️⃣  MODEL TRAINING SUMMARY")
print("="*60)

print(f"""
📌 FINAL RESULTS:

✅ Best Model: {best_model_name}
   • Accuracy: {tuned_accuracy:.4f}
   • Improvement after tuning: {tuned_accuracy - best_accuracy:+.4f}

📊 Model Performance:
   • Correctly identified {tn} pure honey samples
   • Correctly identified {tp} adulterated honey samples
   • Missed {fn} adulterated samples (False Negatives)
   • Flagged {fp} pure samples as adulterated (False Positives)

📁 Saved Files:
   • honey_best_model.pkl - Trained model for deployment
   • model_metadata.json - Model information
   • honey_eda_plots/09_model_training_results.png - Visualizations

🎯 Next Steps:
   • Step 8: Model Evaluation (Detailed analysis)
   • Step 9: Create prediction interface
   • Step 10: Deploy model as API
""")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)