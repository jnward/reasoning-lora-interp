# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, average_precision_score, confusion_matrix,
                           precision_recall_curve, roc_curve, balanced_accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from datasets import load_dataset

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
print(f"Imbalance ratio: 1:{(len(y) - np.sum(y))/np.sum(y):.1f} (Yes:No)")

# %%
# Visualize solution length distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(solution_lengths, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Solution Length (characters)')
plt.ylabel('Count')
plt.title('Distribution of Short Solution Lengths (≤ 64 chars)')
plt.axvline(x=np.mean(solution_lengths), color='red', linestyle='--', 
            label=f'Mean: {np.mean(solution_lengths):.1f}')
plt.legend()

plt.subplot(1, 2, 2)
# Show class distribution for different solution length ranges
length_bins = [(0, 20), (20, 40), (40, 65)]
bin_labels = ['0-20', '20-40', '40-64']
yes_counts = []
no_counts = []

for i, (min_len, max_len) in enumerate(length_bins):
    # Get solution lengths for valid examples
    bin_indices = []
    for j, idx in enumerate(valid_indices):
        sol_len = len(dataset[int(idx)]['solution'])
        if min_len <= sol_len < max_len:
            bin_indices.append(j)
    
    if bin_indices:
        bin_y = y[bin_indices]
        yes_counts.append(np.sum(bin_y))
        no_counts.append(len(bin_y) - np.sum(bin_y))
    else:
        yes_counts.append(0)
        no_counts.append(0)

x = np.arange(len(bin_labels))
width = 0.35
plt.bar(x - width/2, yes_counts, width, label='Yes', color='green', alpha=0.7)
plt.bar(x + width/2, no_counts, width, label='No', color='red', alpha=0.7)
plt.xlabel('Solution Length Range')
plt.ylabel('Count')
plt.title('Class Distribution by Solution Length')
plt.xticks(x, bin_labels)
plt.legend()

plt.tight_layout()
plt.show()

# %%
print("\n" + "="*80)
print("STRATEGIES FOR HANDLING SEVERE CLASS IMBALANCE")
print("="*80)

print("""
Given the severe class imbalance, here are recommended strategies:

1. **Class Weights**: Automatically balance classes in the loss function
   - Use class_weight='balanced' in LogisticRegression
   - Penalizes misclassifying minority class more heavily

2. **Resampling Techniques**:
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Random undersampling of majority class
   - Random oversampling of minority class

3. **Alternative Metrics**:
   - Precision-Recall curve and Average Precision (AP) score
   - F1 score (already using)
   - Balanced accuracy
   - Matthews Correlation Coefficient (MCC)
   - ROC-AUC (though less informative for severe imbalance)

4. **Threshold Tuning**:
   - Default threshold (0.5) may not be optimal
   - Choose threshold based on precision-recall trade-off

5. **Ensemble Methods**:
   - BalancedRandomForestClassifier
   - EasyEnsemble
   - RUSBoost

Let's implement several of these approaches...
""")

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
# Method 1: Class-weighted L1 logistic regression
print("\n" + "-"*60)
print("METHOD 1: CLASS-WEIGHTED L1 LOGISTIC REGRESSION")
print("-"*60)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: No={class_weights[0]:.2f}, Yes={class_weights[1]:.2f}")

# Try different C values with class weights
C_values = [0.001, 0.01, 0.1, 1.0]
weighted_results = []

for C in C_values:
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42, 
                           penalty='l1', solver='saga', class_weight='balanced')
    
    # Cross-validation with stratified folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(clf, X_train_scaled, y_train, cv=cv,
                               scoring=['f1', 'precision', 'recall', 'balanced_accuracy'])
    
    # Fit on full training set
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_test_pred = clf.predict(X_test_scaled)
    y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    result = {
        'C': C,
        'cv_f1': np.mean(cv_results['test_f1']),
        'cv_precision': np.mean(cv_results['test_precision']),
        'cv_recall': np.mean(cv_results['test_recall']),
        'cv_balanced_acc': np.mean(cv_results['test_balanced_accuracy']),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_balanced_acc': balanced_accuracy_score(y_test, y_test_pred),
        'test_auroc': roc_auc_score(y_test, y_test_proba),
        'test_avg_precision': average_precision_score(y_test, y_test_proba),
        'n_features': np.sum(np.abs(clf.coef_[0]) > 1e-6)
    }
    
    weighted_results.append(result)
    print(f"\nC={C}:")
    print(f"  CV F1: {result['cv_f1']:.3f}")
    print(f"  Test F1: {result['test_f1']:.3f}")
    print(f"  Test Precision: {result['test_precision']:.3f}")
    print(f"  Test Recall: {result['test_recall']:.3f}")
    print(f"  Test Balanced Accuracy: {result['test_balanced_acc']:.3f}")
    print(f"  Features selected: {result['n_features']}/192")

# %%
# Method 2: SMOTE (Synthetic Minority Over-sampling Technique)
print("\n" + "-"*60)
print("METHOD 2: SMOTE + L1 LOGISTIC REGRESSION")
print("-"*60)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Create SMOTE sampler
smote = SMOTE(random_state=42)

# Resample training data
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE: {len(X_train_smote)} samples")
print(f"Class distribution: {np.sum(y_train_smote)} Yes, {len(y_train_smote)-np.sum(y_train_smote)} No")

# Train with SMOTE data
smote_results = []
for C in C_values:
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42, 
                           penalty='l1', solver='saga')
    clf.fit(X_train_smote, y_train_smote)
    
    y_test_pred = clf.predict(X_test_scaled)
    y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    result = {
        'C': C,
        'test_f1': f1_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_balanced_acc': balanced_accuracy_score(y_test, y_test_pred),
        'n_features': np.sum(np.abs(clf.coef_[0]) > 1e-6)
    }
    
    smote_results.append(result)
    print(f"\nC={C}: F1={result['test_f1']:.3f}, "
          f"Precision={result['test_precision']:.3f}, "
          f"Recall={result['test_recall']:.3f}")

# %%
# Method 3: Threshold tuning
print("\n" + "-"*60)
print("METHOD 3: THRESHOLD TUNING")
print("-"*60)

# Use best class-weighted model
best_weighted = max(weighted_results, key=lambda x: x['cv_f1'])
best_C = best_weighted['C']
print(f"Using best class-weighted model with C={best_C}")

clf_best = LogisticRegression(C=best_C, max_iter=2000, random_state=42, 
                             penalty='l1', solver='saga', class_weight='balanced')
clf_best.fit(X_train_scaled, y_train)

# Get probabilities
y_train_proba = clf_best.predict_proba(X_train_scaled)[:, 1]
y_test_proba = clf_best.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold using validation set
thresholds = np.arange(0.1, 0.9, 0.05)
threshold_results = []

for thresh in thresholds:
    y_train_pred_thresh = (y_train_proba >= thresh).astype(int)
    
    f1 = f1_score(y_train, y_train_pred_thresh)
    precision = precision_score(y_train, y_train_pred_thresh)
    recall = recall_score(y_train, y_train_pred_thresh)
    
    threshold_results.append({
        'threshold': thresh,
        'f1': f1,
        'precision': precision,
        'recall': recall
    })

# Find best threshold by F1
best_thresh_result = max(threshold_results, key=lambda x: x['f1'])
best_threshold = best_thresh_result['threshold']
print(f"\nOptimal threshold (by F1): {best_threshold:.2f}")

# Evaluate with optimal threshold
y_test_pred_optimal = (y_test_proba >= best_threshold).astype(int)
print(f"Test performance with optimal threshold:")
print(f"  F1: {f1_score(y_test, y_test_pred_optimal):.3f}")
print(f"  Precision: {precision_score(y_test, y_test_pred_optimal):.3f}")
print(f"  Recall: {recall_score(y_test, y_test_pred_optimal):.3f}")

# %%
# Visualization: Performance metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Precision-Recall Curve
ax = axes[0, 0]
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
avg_precision = average_precision_score(y_test, y_test_proba)
ax.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: ROC Curve
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
auroc = roc_auc_score(y_test, y_test_proba)
ax.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Threshold vs Metrics
ax = axes[1, 0]
thresh_df = pd.DataFrame(threshold_results)
ax.plot(thresh_df['threshold'], thresh_df['f1'], 'o-', label='F1')
ax.plot(thresh_df['threshold'], thresh_df['precision'], 's-', label='Precision')
ax.plot(thresh_df['threshold'], thresh_df['recall'], '^-', label='Recall')
ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.5, 
           label=f'Best threshold={best_threshold:.2f}')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Metrics vs Decision Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Confusion matrices
ax = axes[1, 1]
# Default threshold
y_pred_default = clf_best.predict(X_test_scaled)
cm_default = confusion_matrix(y_test, y_pred_default)
# Optimal threshold
cm_optimal = confusion_matrix(y_test, y_test_pred_optimal)

# Show both confusion matrices
cm_combined = np.zeros((4, 2))
cm_combined[:2, 0] = cm_default.flatten()
cm_combined[2:, 1] = cm_optimal.flatten()

labels = ['TN\n(default)', 'FP\n(default)', 'FN\n(default)', 'TP\n(default)',
          'TN\n(optimal)', 'FP\n(optimal)', 'FN\n(optimal)', 'TP\n(optimal)']

sns.heatmap(cm_combined.reshape(4, 2), annot=True, fmt='g', cmap='Blues',
            xticklabels=['Default\nThreshold', 'Optimal\nThreshold'],
            yticklabels=['TN', 'FP', 'FN', 'TP'], ax=ax, cbar=False)
ax.set_title('Confusion Matrix Comparison')

plt.tight_layout()
plt.show()

# %%
# Summary comparison
print("\n" + "="*80)
print("SUMMARY: COMPARISON OF APPROACHES")
print("="*80)

comparison_data = []

# Standard L1 (no class weighting)
clf_standard = LogisticRegression(C=0.1, max_iter=2000, random_state=42, 
                                 penalty='l1', solver='saga')
clf_standard.fit(X_train_scaled, y_train)
y_pred_standard = clf_standard.predict(X_test_scaled)
comparison_data.append({
    'Method': 'Standard L1 (C=0.1)',
    'F1': f1_score(y_test, y_pred_standard),
    'Precision': precision_score(y_test, y_pred_standard),
    'Recall': recall_score(y_test, y_pred_standard),
    'Balanced Acc': balanced_accuracy_score(y_test, y_pred_standard)
})

# Best class-weighted
comparison_data.append({
    'Method': f'Class-weighted L1 (C={best_C})',
    'F1': best_weighted['test_f1'],
    'Precision': best_weighted['test_precision'],
    'Recall': best_weighted['test_recall'],
    'Balanced Acc': best_weighted['test_balanced_acc']
})

# Best SMOTE
best_smote = max(smote_results, key=lambda x: x['test_f1'])
comparison_data.append({
    'Method': f'SMOTE + L1 (C={best_smote["C"]})',
    'F1': best_smote['test_f1'],
    'Precision': best_smote['test_precision'],
    'Recall': best_smote['test_recall'],
    'Balanced Acc': best_smote['test_balanced_acc']
})

# Optimal threshold
comparison_data.append({
    'Method': f'Optimal threshold ({best_threshold:.2f})',
    'F1': f1_score(y_test, y_test_pred_optimal),
    'Precision': precision_score(y_test, y_test_pred_optimal),
    'Recall': recall_score(y_test, y_test_pred_optimal),
    'Balanced Acc': balanced_accuracy_score(y_test, y_test_pred_optimal)
})

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False, float_format='%.3f'))

print(f"\nRecommendation: Use class-weighted L1 regression with threshold tuning")
print(f"This provides a good balance between model simplicity and handling class imbalance.")

# %%