# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
import pandas as pd

# %%
# Load the cached features
print("Loading cached features...")

# Load the summary features
cache_data = np.load('lora_activation_cache/summary_activations_n1000_last16_r1.npz')
X = cache_data['X']
y = cache_data['y']
valid_indices = cache_data['valid_indices']

print(f"Loaded {len(X)} examples")
print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {np.sum(y)} Yes, {len(y) - np.sum(y)} No")

# %%
# Split and scale data
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, valid_indices, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {len(X_train)} examples")
print(f"Test set: {len(X_test)} examples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Train with C=0.07
C = 0.07
print(f"\nTraining L1 logistic regression with C={C}")

clf = LogisticRegression(C=C, max_iter=2000, random_state=42, penalty='l1', solver='saga')
clf.fit(X_train_scaled, y_train)

# Evaluate performance
train_acc = clf.score(X_train_scaled, y_train)
test_acc = clf.score(X_test_scaled, y_test)
y_train_pred = clf.predict(X_train_scaled)
y_test_pred = clf.predict(X_test_scaled)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print(f"\nPerformance:")
print(f"Train F1 score: {train_f1:.4f}")
print(f"Test F1 score: {test_f1:.4f}")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test precision: {test_precision:.4f}")
print(f"Test recall: {test_recall:.4f}")
print(f"Train-test F1 gap: {train_f1 - test_f1:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Incorrect', 'Correct']))

# Cross-validation
from sklearn.model_selection import cross_val_score, cross_validate
print("\n5-fold Cross-Validation (Full Model):")
cv_scores = cross_validate(
    LogisticRegression(C=C, max_iter=2000, penalty='l1', solver='saga'),
    X_train_scaled, y_train, cv=5, 
    scoring=['accuracy', 'f1', 'precision', 'recall']
)
print(f"CV F1 scores: {cv_scores['test_f1']}")
print(f"Mean CV F1: {np.mean(cv_scores['test_f1']):.3f} (+/- {np.std(cv_scores['test_f1'])*2:.3f})")
print(f"Mean CV accuracy: {np.mean(cv_scores['test_accuracy']):.3f} (+/- {np.std(cv_scores['test_accuracy'])*2:.3f})")

# %%
# Extract and analyze features
coefficients = clf.coef_[0]
non_zero_mask = np.abs(coefficients) > 1e-6
non_zero_indices = np.where(non_zero_mask)[0]
non_zero_coefs = coefficients[non_zero_mask]

print(f"\nSelected {len(non_zero_indices)} out of 192 features ({len(non_zero_indices)/192*100:.1f}%)")

# %%
# Create detailed feature list
n_layers = 64
proj_types = ['gate_proj', 'up_proj', 'down_proj']

# Create a list of all features with their information
feature_list = []
for idx, coef in zip(non_zero_indices, non_zero_coefs):
    proj_idx = idx // n_layers
    layer_idx = idx % n_layers
    proj_type = proj_types[proj_idx]
    
    feature_list.append({
        'feature_idx': idx,
        'projection': proj_type,
        'layer': layer_idx,
        'coefficient': coef,
        'abs_coefficient': abs(coef),
        'direction': 'positive' if coef > 0 else 'negative'
    })

# Convert to DataFrame and sort by absolute coefficient
features_df = pd.DataFrame(feature_list)
features_df = features_df.sort_values('abs_coefficient', ascending=False)

# %%
# Display all selected features
print("\n" + "="*80)
print("ALL SELECTED FEATURES (sorted by importance)")
print("="*80)
print(f"{'Rank':<6} {'Projection':<12} {'Layer':<7} {'Coefficient':<12} {'Direction':<10} {'Abs Value':<10}")
print("-"*80)

for i, row in enumerate(features_df.itertuples(), 1):
    print(f"{i:<6} {row.projection:<12} {row.layer:<7} {row.coefficient:>11.6f} {row.direction:<10} {row.abs_coefficient:>9.6f}")

# %%
# Summary statistics by projection type
print("\n" + "="*80)
print("SUMMARY BY PROJECTION TYPE")
print("="*80)

for proj_type in proj_types:
    proj_features = features_df[features_df['projection'] == proj_type]
    
    if len(proj_features) > 0:
        print(f"\n{proj_type.upper()}:")
        print(f"  Total selected: {len(proj_features)} layers")
        print(f"  Layers: {sorted(proj_features['layer'].tolist())}")
        print(f"  Mean |coefficient|: {proj_features['abs_coefficient'].mean():.4f}")
        print(f"  Max |coefficient|: {proj_features['abs_coefficient'].max():.4f}")
        print(f"  Positive coefficients: {(proj_features['coefficient'] > 0).sum()}")
        print(f"  Negative coefficients: {(proj_features['coefficient'] < 0).sum()}")
    else:
        print(f"\n{proj_type.upper()}: No features selected")

# %%
# Visualize selected features
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Bar plot of top 20 features
ax = axes[0, 0]
top_20 = features_df.head(20)
colors = ['green' if x > 0 else 'red' for x in top_20['coefficient']]
bars = ax.barh(range(len(top_20)), top_20['abs_coefficient'], color=colors)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels([f"{row.projection} L{row.layer}" for row in top_20.itertuples()])
ax.set_xlabel('|Coefficient|')
ax.set_title('Top 20 Most Important Features')
ax.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_20['coefficient'])):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', va='center', fontsize=8)

# Plot 2: Distribution of coefficients
ax = axes[0, 1]
ax.hist(features_df['coefficient'], bins=20, edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Coefficient Value')
ax.set_ylabel('Count')
ax.set_title('Distribution of Non-zero Coefficients')

# Plot 3: Features by layer
ax = axes[1, 0]
for i, proj_type in enumerate(proj_types):
    proj_features = features_df[features_df['projection'] == proj_type]
    layers = proj_features['layer'].values
    coefs = proj_features['coefficient'].values
    
    # Plot positive and negative separately
    pos_mask = coefs > 0
    if pos_mask.any():
        ax.scatter(layers[pos_mask], [i]*sum(pos_mask), 
                  s=np.abs(coefs[pos_mask])*1000, 
                  c='green', alpha=0.6, label=f'{proj_type} (+)' if i == 0 else '')
    if (~pos_mask).any():
        ax.scatter(layers[~pos_mask], [i]*sum(~pos_mask), 
                  s=np.abs(coefs[~pos_mask])*1000, 
                  c='red', alpha=0.6, label=f'{proj_type} (-)' if i == 0 else '')

ax.set_xlabel('Layer')
ax.set_yticks(range(3))
ax.set_yticklabels(proj_types)
ax.set_title('Selected Features by Layer (size = |coefficient|)')
ax.set_xlim(-1, 64)
ax.grid(True, alpha=0.3)
if ax.get_legend_handles_labels()[0]:  # Only add legend if there are items
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 4: Heatmap of selected features
ax = axes[1, 1]
selection_matrix = np.zeros((3, n_layers))
for _, row in features_df.iterrows():
    proj_idx = proj_types.index(row['projection'])
    selection_matrix[proj_idx, row['layer']] = row['coefficient']

im = ax.imshow(selection_matrix, aspect='auto', cmap='RdBu_r', 
               vmin=-features_df['abs_coefficient'].max(), 
               vmax=features_df['abs_coefficient'].max())
ax.set_yticks(range(3))
ax.set_yticklabels(proj_types)
ax.set_xlabel('Layer')
ax.set_title('Coefficient Heatmap')
plt.colorbar(im, ax=ax, label='Coefficient')

plt.tight_layout()
plt.show()

# %%
# Analyze layer distribution
print("\n" + "="*80)
print("LAYER DISTRIBUTION ANALYSIS")
print("="*80)

# Count features per layer range
layer_ranges = {
    'Early (0-15)': (0, 16),
    'Mid-Early (16-31)': (16, 32),
    'Mid-Late (32-47)': (32, 48),
    'Late (48-63)': (48, 64)
}

for range_name, (start, end) in layer_ranges.items():
    range_features = features_df[(features_df['layer'] >= start) & (features_df['layer'] < end)]
    print(f"\n{range_name}:")
    print(f"  Total features: {len(range_features)}")
    print(f"  Mean |coefficient|: {range_features['abs_coefficient'].mean():.4f}" if len(range_features) > 0 else "  No features")
    
    for proj_type in proj_types:
        proj_range = range_features[range_features['projection'] == proj_type]
        if len(proj_range) > 0:
            print(f"  {proj_type}: {len(proj_range)} features")

# %%
# Export feature list to CSV
output_file = 'selected_features_c007.csv'
features_df.to_csv(output_file, index=False)
print(f"\nFeature list saved to: {output_file}")

# %%
# Create a summary report
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)
print(f"Regularization parameter C: {C}")
print(f"Total features selected: {len(features_df)}/192 ({len(features_df)/192*100:.1f}%)")
print(f"Test F1 score: {test_f1:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Mean CV F1: {np.mean(cv_scores['test_f1']):.3f}")
print(f"Positive coefficients: {(features_df['coefficient'] > 0).sum()}")
print(f"Negative coefficients: {(features_df['coefficient'] < 0).sum()}")
print(f"\nMost important feature: {features_df.iloc[0]['projection']} Layer {features_df.iloc[0]['layer']} (coef={features_df.iloc[0]['coefficient']:.4f})")
print(f"Strongest positive: {features_df[features_df['coefficient'] > 0].iloc[0]['projection']} Layer {features_df[features_df['coefficient'] > 0].iloc[0]['layer']}")
print(f"Strongest negative: {features_df[features_df['coefficient'] < 0].iloc[0]['projection']} Layer {features_df[features_df['coefficient'] < 0].iloc[0]['layer']}")

# %%
# Train classifier using only layer 38 up_proj
print("\n" + "="*80)
print("SINGLE FEATURE CLASSIFIER: Layer 38 up_proj")
print("="*80)

# Extract only layer 38 up_proj feature
# up_proj starts at index 64 (after gate_proj), so layer 38 is at index 64 + 38 = 102
layer_38_up_proj_idx = 1 * n_layers + 38  # 1 for up_proj, 38 for layer
print(f"Feature index for layer 38 up_proj: {layer_38_up_proj_idx}")

# Extract single feature
X_single = X[:, layer_38_up_proj_idx].reshape(-1, 1)
X_train_single = X_single[idx_train]
X_test_single = X_single[idx_test]

# Scale the single feature
scaler_single = StandardScaler()
X_train_single_scaled = scaler_single.fit_transform(X_train_single)
X_test_single_scaled = scaler_single.transform(X_test_single)

# Train logistic regression with single feature
print("\nTraining with single feature...")
clf_single = LogisticRegression(max_iter=1000, random_state=42)
clf_single.fit(X_train_single_scaled, y_train)

# Evaluate
train_acc_single = clf_single.score(X_train_single_scaled, y_train)
test_acc_single = clf_single.score(X_test_single_scaled, y_test)
y_train_pred_single = clf_single.predict(X_train_single_scaled)
y_test_pred_single = clf_single.predict(X_test_single_scaled)
train_f1_single = f1_score(y_train, y_train_pred_single)
test_f1_single = f1_score(y_test, y_test_pred_single)
test_precision_single = precision_score(y_test, y_test_pred_single)
test_recall_single = recall_score(y_test, y_test_pred_single)

print(f"\nSingle Feature Performance:")
print(f"Train F1 score: {train_f1_single:.4f}")
print(f"Test F1 score: {test_f1_single:.4f}")
print(f"Train accuracy: {train_acc_single:.4f}")
print(f"Test accuracy: {test_acc_single:.4f}")
print(f"Test precision: {test_precision_single:.4f}")
print(f"Test recall: {test_recall_single:.4f}")
print(f"Coefficient: {clf_single.coef_[0][0]:.4f}")
print(f"Intercept: {clf_single.intercept_[0]:.4f}")

# Compare with full model
print(f"\nComparison:")
print(f"Full model (192 features) test F1: {test_f1:.4f}")
print(f"Single feature (layer 38 up_proj) test F1: {test_f1_single:.4f}")
print(f"F1 retained: {test_f1_single/test_f1*100:.1f}%")

# %%
# Visualize the single feature's discriminative power
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Distribution of layer 38 up_proj activation by class
ax1.hist(X_train_single[y_train == 0], bins=30, alpha=0.5, label='Incorrect', color='red', density=True)
ax1.hist(X_train_single[y_train == 1], bins=30, alpha=0.5, label='Correct', color='green', density=True)
ax1.set_xlabel('Layer 38 up_proj Activation')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of Layer 38 up_proj Activation by Answer Correctness')
ax1.legend()

# Plot 2: Decision boundary visualization
# Create a range of values for the feature
feature_range = np.linspace(X_train_single_scaled.min() - 1, X_train_single_scaled.max() + 1, 100)
# Get probabilities for each value
probs = clf_single.predict_proba(feature_range.reshape(-1, 1))[:, 1]

ax2.plot(feature_range, probs, 'b-', linewidth=2, label='P(Correct)')
ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision boundary')
ax2.scatter(X_test_single_scaled[y_test == 0], [0.1] * sum(y_test == 0), 
           color='red', alpha=0.5, label='Test: Incorrect')
ax2.scatter(X_test_single_scaled[y_test == 1], [0.9] * sum(y_test == 1), 
           color='green', alpha=0.5, label='Test: Correct')
ax2.set_xlabel('Layer 38 up_proj Activation (scaled)')
ax2.set_ylabel('Probability')
ax2.set_title('Logistic Regression Decision Function')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Analyze what makes layer 38 up_proj special
print("\n" + "="*80)
print("ANALYSIS: Why is layer 38 up_proj important?")
print("="*80)

# Check if it was selected in the L1 model
if layer_38_up_proj_idx in non_zero_indices:
    idx_in_selected = np.where(non_zero_indices == layer_38_up_proj_idx)[0][0]
    coef_in_l1 = non_zero_coefs[idx_in_selected]
    rank_in_l1 = features_df[features_df['feature_idx'] == layer_38_up_proj_idx].index[0] + 1
    print(f"Layer 38 up_proj in L1 model:")
    print(f"  - Rank: {rank_in_l1}/{len(non_zero_indices)}")
    print(f"  - Coefficient: {coef_in_l1:.4f}")
    print(f"  - Direction: {'positive' if coef_in_l1 > 0 else 'negative'}")
else:
    print("Layer 38 up_proj was NOT selected by L1 regularization!")

# Basic statistics
print(f"\nFeature statistics:")
print(f"  - Mean activation (train): {X_train_single.mean():.4f}")
print(f"  - Std activation (train): {X_train_single.std():.4f}")
print(f"  - Mean activation (correct answers): {X_train_single[y_train == 1].mean():.4f}")
print(f"  - Mean activation (incorrect answers): {X_train_single[y_train == 0].mean():.4f}")
print(f"  - Difference: {X_train_single[y_train == 1].mean() - X_train_single[y_train == 0].mean():.4f}")

# %%
# Cross-validation for single feature
print("\n5-fold Cross-Validation (Single Feature):")
cv_scores_single = cross_validate(
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_single_scaled, y_train, cv=5,
    scoring=['accuracy', 'f1', 'precision', 'recall']
)
print(f"CV F1 scores: {cv_scores_single['test_f1']}")
print(f"Mean CV F1: {np.mean(cv_scores_single['test_f1']):.3f} (+/- {np.std(cv_scores_single['test_f1'])*2:.3f})")
print(f"Mean CV accuracy: {np.mean(cv_scores_single['test_accuracy']):.3f} (+/- {np.std(cv_scores_single['test_accuracy'])*2:.3f})")

print("\nConclusion: Layer 38 up_proj alone achieves {:.1f}% of the full model's F1 performance!".format(test_f1_single/test_f1*100))
# %%
