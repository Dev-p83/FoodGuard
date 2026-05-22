# Step 6: Honey Dataset - Feature Selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🍯 HONEY DATASET - FEATURE SELECTION 🍯")
print("="*60)

# Load the dataset with engineered features
honey_df = pd.read_csv('honey_dataset_engineered.csv')
print("\n📊 Dataset loaded:")
print(f"  Shape: {honey_df.shape}")
print(f"  Total features: {honey_df.shape[1] - 1} (excluding target)")

# Separate features and target
X = honey_df.drop('Is_Adulterated', axis=1)
y = honey_df['Is_Adulterated']

feature_names = X.columns.tolist()
print(f"\n📋 Available features: {len(feature_names)}")

# ============================================================================
# 1. CORRELATION-BASED SELECTION (Remove highly correlated features)
# ============================================================================
print("\n" + "="*60)
print("1️⃣  CORRELATION-BASED SELECTION")
print("="*60)

# Calculate correlation matrix
correlation_matrix = X.corr()

# Find highly correlated features (correlation > 0.85)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.85:
            high_corr_pairs.append({
                'feature1': correlation_matrix.columns[i],
                'feature2': correlation_matrix.columns[j],
                'correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    print("\n⚠️ Highly correlated feature pairs (correlation > 0.85):")
    for pair in high_corr_pairs:
        print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
else:
    print("\n✅ No highly correlated features found (all < 0.85)")

# ============================================================================
# 2. UNIVARIATE FEATURE SELECTION (F-test and Mutual Information)
# ============================================================================
print("\n" + "="*60)
print("2️⃣  UNIVARIATE FEATURE SELECTION")
print("="*60)

# Method 1: ANOVA F-test
f_selector = SelectKBest(score_func=f_classif, k='all')
f_selector.fit(X, y)
f_scores = pd.DataFrame({
    'feature': feature_names,
    'f_score': f_selector.scores_,
    'p_value': f_selector.pvalues_
}).sort_values('f_score', ascending=False)

# Method 2: Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_names,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\n📊 Top 10 Features by F-Score:")
for i, row in f_scores.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:30s} | F-Score: {row['f_score']:.2f} | p-value: {row['p_value']:.4f}")

print("\n📊 Top 10 Features by Mutual Information:")
for i, row in mi_df.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:30s} | MI Score: {row['mi_score']:.4f}")

# ============================================================================
# 3. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*60)
print("3️⃣  RANDOM FOREST FEATURE IMPORTANCE")
print("="*60)

# Train Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

rf_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🌲 Top 10 Features by Random Forest Importance:")
for i, row in rf_importance.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:30s} | Importance: {row['importance']:.4f}")

# ============================================================================
# 4. LASSO REGRESSION (L1 regularization for feature selection)
# ============================================================================
print("\n" + "="*60)
print("4️⃣  LASSO REGRESSION FEATURE SELECTION")
print("="*60)

# Scale features for Lasso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso with cross-validation
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y)

lasso_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lasso.coef_
}).sort_values('coefficient', ascending=False)

# Features with non-zero coefficients (selected by Lasso)
selected_by_lasso = lasso_importance[abs(lasso_importance['coefficient']) > 0]
print(f"\n✓ Lasso selected {len(selected_by_lasso)} features out of {len(feature_names)}")
print("\n📊 Top 10 Features by Lasso Coefficient (absolute value):")
for i, row in selected_by_lasso.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:30s} | Coefficient: {row['coefficient']:.4f}")

# ============================================================================
# 5. RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================================================
print("\n" + "="*60)
print("5️⃣  RECURSIVE FEATURE ELIMINATION (RFE)")
print("="*60)

# Use RFE with Random Forest
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), 
          n_features_to_select=10, step=2)
rfe.fit(X, y)

rfe_selected = pd.DataFrame({
    'feature': feature_names,
    'selected': rfe.support_,
    'ranking': rfe.ranking_
}).sort_values('ranking')

print("\n🎯 Top 10 Features selected by RFE:")
selected_count = 0
for i, row in rfe_selected.iterrows():
    if row['selected']:
        selected_count += 1
        print(f"  {selected_count:2d}. {row['feature']:30s} | Ranking: {row['ranking']}")

# ============================================================================
# 6. CONSENSUS FEATURE SELECTION (Combine all methods)
# ============================================================================
print("\n" + "="*60)
print("6️⃣  CONSENSUS FEATURE SELECTION (Voting Method)")
print("="*60)

# Create a consensus score (0-5 based on how many methods selected it)
consensus_scores = {}

for feature in feature_names:
    score = 0
    
    # Method 1: F-test (top 10)
    if feature in f_scores.head(10)['feature'].values:
        score += 1
    
    # Method 2: Mutual Information (top 10)
    if feature in mi_df.head(10)['feature'].values:
        score += 1
    
    # Method 3: Random Forest (top 10)
    if feature in rf_importance.head(10)['feature'].values:
        score += 1
    
    # Method 4: Lasso (non-zero coefficient)
    if feature in selected_by_lasso['feature'].values:
        score += 1
    
    # Method 5: RFE (selected)
    if feature in rfe_selected[rfe_selected['selected'] == True]['feature'].values:
        score += 1
    
    consensus_scores[feature] = score

consensus_df = pd.DataFrame({
    'feature': list(consensus_scores.keys()),
    'votes': list(consensus_scores.values())
}).sort_values('votes', ascending=False)

print("\n📊 Consensus Voting Results (5 methods total):")
for i, row in consensus_df.head(15).iterrows():
    stars = "⭐" * row['votes']
    print(f"  {row['feature']:30s} | Votes: {row['votes']}/5 {stars}")

# ============================================================================
# 7. SELECT FINAL FEATURES
# ============================================================================
print("\n" + "="*60)
print("7️⃣  SELECTING FINAL FEATURE SET")
print("="*60)

# Select features with at least 3 votes (majority)
final_features = consensus_df[consensus_df['votes'] >= 3]['feature'].tolist()
print(f"\n✅ Final selected features ({len(final_features)} features):")
for i, feature in enumerate(final_features, 1):
    votes = consensus_df[consensus_df['feature'] == feature]['votes'].values[0]
    print(f"  {i:2d}. {feature:30s} (voted by {votes}/5 methods)")

# Features to drop
dropped_features = [f for f in feature_names if f not in final_features]
print(f"\n🗑️ Dropped features ({len(dropped_features)} features):")
for feature in dropped_features[:10]:  # Show first 10
    print(f"  - {feature}")
if len(dropped_features) > 10:
    print(f"  ... and {len(dropped_features) - 10} more")

# ============================================================================
# 8. VISUALIZE FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*60)
print("8️⃣  VISUALIZING FEATURE SELECTION RESULTS")
print("="*60)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Random Forest Feature Importance
ax1 = axes[0, 0]
top10_rf = rf_importance.head(10)
ax1.barh(top10_rf['feature'], top10_rf['importance'], color='skyblue')
ax1.set_xlabel('Importance')
ax1.set_title('Random Forest Feature Importance', fontweight='bold')
ax1.invert_yaxis()

# Plot 2: F-Scores
ax2 = axes[0, 1]
top10_f = f_scores.head(10)
ax2.barh(top10_f['feature'], top10_f['f_score'], color='lightcoral')
ax2.set_xlabel('F-Score')
ax2.set_title('ANOVA F-Test Scores', fontweight='bold')
ax2.invert_yaxis()

# Plot 3: Mutual Information Scores
ax3 = axes[1, 0]
top10_mi = mi_df.head(10)
ax3.barh(top10_mi['feature'], top10_mi['mi_score'], color='lightgreen')
ax3.set_xlabel('Mutual Information Score')
ax3.set_title('Mutual Information Scores', fontweight='bold')
ax3.invert_yaxis()

# Plot 4: Consensus Votes
ax4 = axes[1, 1]
top15_consensus = consensus_df.head(15)
colors = ['gold' if v >= 4 else 'silver' if v >= 3 else '#ffcccc' for v in top15_consensus['votes']]
ax4.barh(top15_consensus['feature'], top15_consensus['votes'], color=colors)
ax4.set_xlabel('Number of Methods Voting (out of 5)')
ax4.set_title('Consensus Feature Selection Results', fontweight='bold')
ax4.invert_yaxis()
ax4.axvline(x=3, color='red', linestyle='--', label='Selection Threshold (3 votes)')
ax4.legend()

plt.suptitle('Honey Dataset: Feature Selection Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('honey_eda_plots/08_feature_selection.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. SAVE FINAL DATASET WITH SELECTED FEATURES
# ============================================================================
print("\n" + "="*60)
print("9️⃣  SAVING FINAL DATASET")
print("="*60)

# Create final dataset with selected features
X_final = X[final_features]
y_final = y

# Save selected features dataset
final_df = pd.concat([X_final, y_final], axis=1)
final_df.to_csv('honey_dataset_final.csv', index=False)

print(f"\n💾 Final dataset saved: 'honey_dataset_final.csv'")
print(f"   Shape: {final_df.shape}")
print(f"   Features: {list(X_final.columns)}")

# Save feature list for future reference
with open('selected_features.txt', 'w') as f:
    f.write("Selected Features for Honey Adulteration Detection\n")
    f.write("="*50 + "\n")
    for feature in final_features:
        f.write(f"{feature}\n")

print("💾 Feature list saved: 'selected_features.txt'")

# ============================================================================
# 10. SUMMARY REPORT
# ============================================================================
print("\n" + "="*60)
print("🔟  FEATURE SELECTION SUMMARY")
print("="*60)

print(f"""
📌 FINAL SELECTION RESULTS:

✅ Selected Features ({len(final_features)}):
   {', '.join(final_features[:5])}...
   
❌ Dropped Features ({len(dropped_features)}):
   {', '.join(dropped_features[:3])}...

📊 Performance Expectation:
   - Reduced dimensionality by {len(feature_names) - len(final_features)} features
   - Removed redundant and noisy features
   - Maintained {len(final_features)} high-quality features

🎯 Next Steps:
   - Proceed to Model Training with these {len(final_features)} features
   - Use train-test split from preprocessing
   - Train multiple models and compare performance
""")

print("\n" + "="*60)
print("✅ HONEY FEATURE SELECTION COMPLETE!")
print("="*60)