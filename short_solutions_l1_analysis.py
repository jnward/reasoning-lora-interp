# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                           balanced_accuracy_score, confusion_matrix,
                           precision_recall_curve, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from datasets import load_dataset
from imblearn.over_sampling import SMOTE

# %%
# Load dataset and filter by solution length
print("Loading s1K-1.1 dataset...")
dataset = load_dataset("simplescaling/s1K-1.1", split="train")

# Filter for short solutions
print("\nFiltering for solutions with ≤ 64 characters...")
short_solution_indices = []
solution_lengths = []

for i, example in enumerate(dataset):
    solution = example.get('solution', '')
    if len(solution) <= 64:
        short_solution_indices.append(i)
        solution_lengths.append(len(solution))

print(f"Found {len(short_solution_indices)} examples with short solutions out of {len(dataset)} total")
print(f"Percentage retained: {len(short_solution_indices)/len(dataset)*100:.1f}%")

# %%
# Load cached features and filter
print("\nLoading cached features...")
cache_data = np.load('lora_activation_cache/summary_activations_n1000_last16_r1.npz')
X_all = cache_data['X']
y_all = cache_data['y']
valid_indices_all = cache_data['valid_indices']

# Find intersection of valid indices and short solution indices
valid_short_indices = [i for i in range(len(valid_indices_all)) 
                      if valid_indices_all[i] in short_solution_indices]

X = X_all[valid_short_indices]
y = y_all[valid_short_indices]
valid_indices = valid_indices_all[valid_short_indices]

print(f"\nAfter filtering:")
print(f"Total examples: {len(X)}")
print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {np.sum(y)} Yes, {len(y) - np.sum(y)} No")
print(f"Class balance: {np.sum(y)/len(y)*100:.1f}% Yes, {(1-np.sum(y)/len(y))*100:.1f}% No")
print(f"Imbalance ratio: 1:{(1-np.sum(y)/np.sum(y)):.1f} (Yes:No)")

# %%
# Split data with stratification
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, valid_indices, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} examples")
print(f"Test set: {len(X_test)} examples")
print(f"Train class distribution: {np.sum(y_train)} Yes ({np.sum(y_train)/len(y_train)*100:.1f}%), {len(y_train)-np.sum(y_train)} No")
print(f"Test class distribution: {np.sum(y_test)} Yes ({np.sum(y_test)/len(y_test)*100:.1f}%), {len(y_test)-np.sum(y_test)} No")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
print("\n" + "="*80)
print("EXPLORING L1 REGULARIZATION WITH CLASS IMBALANCE")
print("="*80)

# Try different C values with and without class weighting
C_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
results_standard = []
results_weighted = []

print("\nComparing standard vs class-weighted L1 regression...")

for C in C_values:
    # Standard L1 (no class weighting)
    clf_std = LogisticRegression(C=C, max_iter=2000, random_state=42, 
                                penalty='l1', solver='saga')
    clf_std.fit(X_train_scaled, y_train)
    
    y_pred_std = clf_std.predict(X_test_scaled)
    
    # Handle case where model predicts all one class
    if len(np.unique(y_pred_std)) == 1:
        f1_std = 0.0
        prec_std = 0.0
        rec_std = 0.0
    else:
        f1_std = f1_score(y_test, y_pred_std)
        prec_std = precision_score(y_test, y_pred_std, zero_division=0)
        rec_std = recall_score(y_test, y_pred_std)
    
    results_standard.append({
        'C': C,
        'f1': f1_std,
        'precision': prec_std,
        'recall': rec_std,
        'n_features': np.sum(np.abs(clf_std.coef_[0]) > 1e-6)
    })
    
    # Class-weighted L1
    clf_wt = LogisticRegression(C=C, max_iter=2000, random_state=42, 
                               penalty='l1', solver='saga', class_weight='balanced')
    clf_wt.fit(X_train_scaled, y_train)
    
    y_pred_wt = clf_wt.predict(X_test_scaled)
    
    # CV for weighted model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        LogisticRegression(C=C, max_iter=2000, random_state=42, 
                          penalty='l1', solver='saga', class_weight='balanced'),
        X_train_scaled, y_train, cv=cv, scoring=['f1', 'precision', 'recall']
    )
    
    results_weighted.append({
        'C': C,
        'f1': f1_score(y_test, y_pred_wt),
        'precision': precision_score(y_test, y_pred_wt, zero_division=0),
        'recall': recall_score(y_test, y_pred_wt),
        'cv_f1': np.mean(cv_results['test_f1']),
        'cv_f1_std': np.std(cv_results['test_f1']),
        'n_features': np.sum(np.abs(clf_wt.coef_[0]) > 1e-6)
    })
    
    print(f"\nC={C}:")
    print(f"  Standard: F1={f1_std:.3f}, Features={results_standard[-1]['n_features']}")
    print(f"  Weighted: F1={results_weighted[-1]['f1']:.3f}, CV F1={results_weighted[-1]['cv_f1']:.3f}±{results_weighted[-1]['cv_f1_std']:.3f}, Features={results_weighted[-1]['n_features']}")

# %%
# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Convert to DataFrames
df_std = pd.DataFrame(results_standard)
df_wt = pd.DataFrame(results_weighted)

# Plot 1: F1 scores
ax = axes[0, 0]
ax.semilogx(df_std['C'], df_std['f1'], 'o-', label='Standard L1', markersize=8)
ax.semilogx(df_wt['C'], df_wt['f1'], 's-', label='Class-weighted L1', markersize=8)
ax.semilogx(df_wt['C'], df_wt['cv_f1'], '^--', label='Class-weighted CV F1', markersize=8)
ax.set_xlabel('C (Regularization Parameter)')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score vs Regularization Strength')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Precision-Recall trade-off
ax = axes[0, 1]
ax.plot(df_std['recall'], df_std['precision'], 'o-', label='Standard L1', markersize=8)
ax.plot(df_wt['recall'], df_wt['precision'], 's-', label='Class-weighted L1', markersize=8)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Trade-off')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Number of features
ax = axes[1, 0]
ax.semilogx(df_std['C'], df_std['n_features'], 'o-', label='Standard L1', markersize=8)
ax.semilogx(df_wt['C'], df_wt['n_features'], 's-', label='Class-weighted L1', markersize=8)
ax.set_xlabel('C (Regularization Parameter)')
ax.set_ylabel('Number of Selected Features')
ax.set_title('Feature Selection vs Regularization')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Class distribution visualization
ax = axes[1, 1]
# Show train and test class distributions
labels = ['Train', 'Test']
yes_counts = [np.sum(y_train), np.sum(y_test)]
no_counts = [len(y_train) - np.sum(y_train), len(y_test) - np.sum(y_test)]

x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, yes_counts, width, label='Yes', color='green', alpha=0.7)
ax.bar(x + width/2, no_counts, width, label='No', color='red', alpha=0.7)
ax.set_ylabel('Count')
ax.set_title('Class Distribution')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add percentage labels
for i, (y_count, n_count) in enumerate(zip(yes_counts, no_counts)):
    total = y_count + n_count
    ax.text(i - width/2, y_count + 1, f'{y_count/total*100:.1f}%', ha='center')
    ax.text(i + width/2, n_count + 1, f'{n_count/total*100:.1f}%', ha='center')

plt.tight_layout()
plt.show()

# %%
# Select best model based on CV F1
best_idx = np.argmax([r['cv_f1'] for r in results_weighted])
best_C = results_weighted[best_idx]['C']
print(f"\nBest C value (by CV F1): {best_C}")
print(f"Performance: {results_weighted[best_idx]}")

# Train final model
clf_final = LogisticRegression(C=best_C, max_iter=2000, random_state=42, 
                              penalty='l1', solver='saga', class_weight='balanced')
clf_final.fit(X_train_scaled, y_train)

# %%
# Analyze selected features
coefficients = clf_final.coef_[0]
non_zero_mask = np.abs(coefficients) > 1e-6
non_zero_indices = np.where(non_zero_mask)[0]
non_zero_coefs = coefficients[non_zero_mask]

print(f"\n{len(non_zero_indices)} features selected with C={best_C}")

# Create feature names
n_layers = 64
proj_types = ['gate_proj', 'up_proj', 'down_proj']
feature_names = []
for proj_type in proj_types:
    for layer in range(n_layers):
        feature_names.append(f'{proj_type}_L{layer}')

# List top features
feature_importance = []
for idx, coef in zip(non_zero_indices, non_zero_coefs):
    feature_importance.append({
        'feature': feature_names[idx],
        'coefficient': coef,
        'abs_coefficient': abs(coef)
    })

feature_df = pd.DataFrame(feature_importance).sort_values('abs_coefficient', ascending=False)

print("\nTop 10 most important features:")
for i, row in enumerate(feature_df.head(10).itertuples(), 1):
    print(f"{i}. {row.feature}: {row.coefficient:.4f}")

# %%
# SMOTE comparison
print("\n" + "="*80)
print("SMOTE COMPARISON")
print("="*80)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE: {len(X_train_smote)} samples")
print(f"Class distribution: {np.sum(y_train_smote)} Yes, {len(y_train_smote)-np.sum(y_train_smote)} No")

# Train with best C on SMOTE data
clf_smote = LogisticRegression(C=best_C, max_iter=2000, random_state=42, 
                               penalty='l1', solver='saga')
clf_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = clf_smote.predict(X_test_scaled)
f1_smote = f1_score(y_test, y_pred_smote)
print(f"\nSMOTE + L1 (C={best_C}): F1={f1_smote:.3f}")

# %%
# Confusion matrices comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Standard L1
clf_std_best = LogisticRegression(C=best_C, max_iter=2000, random_state=42, 
                                 penalty='l1', solver='saga')
clf_std_best.fit(X_train_scaled, y_train)
y_pred_std = clf_std_best.predict(X_test_scaled)
cm_std = confusion_matrix(y_test, y_pred_std)

# Class-weighted
y_pred_wt = clf_final.predict(X_test_scaled)
cm_wt = confusion_matrix(y_test, y_pred_wt)

# SMOTE
cm_smote = confusion_matrix(y_test, y_pred_smote)

# Plot confusion matrices
for ax, cm, title in zip(axes, [cm_std, cm_wt, cm_smote], 
                         ['Standard L1', 'Class-weighted L1', 'SMOTE + L1']):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    ax.set_title(f'{title} (C={best_C})')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Add metrics
    tn, fp, fn, tp = cm.ravel()
    f1 = 2*tp/(2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    precision = tp/(tp + fp) if (tp + fp) > 0 else 0
    recall = tp/(tp + fn) if (tp + fn) > 0 else 0
    
    ax.text(0.5, -0.15, f'F1={f1:.3f}, Prec={precision:.3f}, Rec={recall:.3f}', 
            transform=ax.transAxes, ha='center')

plt.tight_layout()
plt.show()

# %%
# Summary recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS FOR SEVERE CLASS IMBALANCE")
print("="*80)

print(f"""
With {np.sum(y)/len(y)*100:.1f}% positive class (Yes) and {(1-np.sum(y)/len(y))*100:.1f}% negative class (No):

1. **Use class-weighted L1 regression** (class_weight='balanced')
   - Best C value: {best_C}
   - Test F1: {results_weighted[best_idx]['f1']:.3f}
   - Features selected: {results_weighted[best_idx]['n_features']}/192

2. **Focus on F1 score and balanced accuracy** rather than raw accuracy
   - Raw accuracy can be misleading with severe imbalance

3. **Consider threshold tuning** for deployment
   - The default 0.5 threshold may not be optimal
   - Choose based on precision-recall requirements

4. **SMOTE can help** but may not always improve performance
   - Creates synthetic minority examples
   - Can lead to overfitting if not careful

5. **Monitor both precision and recall**
   - High recall: Finding all "No" answers (minority class)
   - High precision: When we predict "No", we're usually right
""")

# %%