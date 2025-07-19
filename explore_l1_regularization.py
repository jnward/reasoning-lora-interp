# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# %%
# Load the cached features
print("Loading cached features...")

# Load the summary features (already computed from full activations)
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
# Explore C values between 0.01 and 0.1
print("\nExploring L1 regularization in the sweet spot (C=0.01 to C=0.1)...")
print("Running 5-fold CV for each C value...\n")

# Import cross-validation function
from sklearn.model_selection import cross_val_score, cross_validate

# Create a finer grid of C values
C_values = np.logspace(np.log10(0.01), np.log10(0.1), 20)
results = []

for C in C_values:
    # Train with L1 regularization
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42, penalty='l1', solver='saga')
    clf.fit(X_train_scaled, y_train)
    
    # Count non-zero coefficients
    non_zero = np.sum(np.abs(clf.coef_[0]) > 1e-6)
    
    # Evaluate on train/test split
    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)
    
    # Get predictions for additional metrics
    y_train_pred = clf.predict(X_train_scaled)
    y_test_pred = clf.predict(X_test_scaled)
    
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    
    # 5-fold cross-validation
    cv_results = cross_validate(
        LogisticRegression(C=C, max_iter=2000, penalty='l1', solver='saga'),
        X_train_scaled, y_train, cv=5, scoring=['f1', 'accuracy']
    )
    cv_f1_mean = np.mean(cv_results['test_f1'])
    cv_f1_std = np.std(cv_results['test_f1'])
    cv_acc_mean = np.mean(cv_results['test_accuracy'])
    
    results.append({
        'C': C,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'cv_f1_mean': cv_f1_mean,
        'cv_f1_std': cv_f1_std,
        'cv_acc_mean': cv_acc_mean,
        'gap': train_f1 - test_f1,  # Now using F1 gap
        'non_zero': non_zero,
        'precision': test_precision,
        'recall': test_recall
    })
    
    print(f"C={C:.4f}: Test Acc={test_acc:.3f}, Test F1={test_f1:.3f}, CV F1={cv_f1_mean:.3f}±{cv_f1_std:.3f}, Features={non_zero}")

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# %%
# Plot the results
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: F1 Score vs C
ax = axes[0, 0]
ax.semilogx(results_df['C'], results_df['train_f1'], 'o-', label='Train F1', markersize=6)
ax.semilogx(results_df['C'], results_df['test_f1'], 's-', label='Test F1', markersize=6)
ax.semilogx(results_df['C'], results_df['cv_f1_mean'], '^-', label='CV F1 (mean)', markersize=6)
# Add error bars for CV
ax.errorbar(results_df['C'], results_df['cv_f1_mean'], 
            yerr=results_df['cv_f1_std']*2, fmt='none', ecolor='gray', alpha=0.5)
ax.set_xlabel('C (Regularization Parameter)')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score vs Regularization Strength')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Number of features vs C
ax = axes[0, 1]
ax.semilogx(results_df['C'], results_df['non_zero'], 'g^-', markersize=6)
ax.set_xlabel('C (Regularization Parameter)')
ax.set_ylabel('Number of Non-zero Features')
ax.set_title('Feature Selection vs Regularization')
ax.grid(True, alpha=0.3)

# Plot 3: Train-Test F1 Gap vs C
ax = axes[1, 0]
ax.semilogx(results_df['C'], results_df['gap'], 'ro-', markersize=6)
ax.set_xlabel('C (Regularization Parameter)')
ax.set_ylabel('Train-Test F1 Gap')
ax.set_title('Overfitting vs Regularization (F1 Score)')
ax.grid(True, alpha=0.3)

# Plot 4: Precision/Recall vs C
ax = axes[1, 1]
ax.semilogx(results_df['C'], results_df['precision'], 'b*-', label='Precision', markersize=8)
ax.semilogx(results_df['C'], results_df['recall'], 'm+-', label='Recall', markersize=8)
ax.set_xlabel('C (Regularization Parameter)')
ax.set_ylabel('Score')
ax.set_title('Precision and Recall vs Regularization')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Accuracy vs C
ax = axes[0, 2]
ax.semilogx(results_df['C'], results_df['train_acc'], 'o-', label='Train Acc', markersize=6)
ax.semilogx(results_df['C'], results_df['test_acc'], 's-', label='Test Acc', markersize=6)
ax.semilogx(results_df['C'], results_df['cv_acc_mean'], '^-', label='CV Acc (mean)', markersize=6)
ax.set_xlabel('C (Regularization Parameter)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Regularization Strength')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: F1 vs Accuracy scatter
ax = axes[1, 2]
ax.scatter(results_df['test_acc'], results_df['test_f1'], s=results_df['non_zero']*2, 
           c=np.log10(results_df['C']), cmap='viridis', alpha=0.7)
ax.set_xlabel('Test Accuracy')
ax.set_ylabel('Test F1 Score')
ax.set_title('F1 vs Accuracy (size = num features)')
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('log10(C)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Find optimal C based on different criteria
print("\nOptimal C values based on different criteria:")

# Best CV F1 (most robust)
best_cv_idx = results_df['cv_f1_mean'].idxmax()
print(f"Best CV F1: C={results_df.loc[best_cv_idx, 'C']:.4f} "
      f"(cv_f1={results_df.loc[best_cv_idx, 'cv_f1_mean']:.3f}±{results_df.loc[best_cv_idx, 'cv_f1_std']:.3f}, "
      f"test_acc={results_df.loc[best_cv_idx, 'test_acc']:.3f}, "
      f"test_f1={results_df.loc[best_cv_idx, 'test_f1']:.3f}, "
      f"features={results_df.loc[best_cv_idx, 'non_zero']})")

# Best test F1
best_test_idx = results_df['test_f1'].idxmax()
print(f"Best test F1: C={results_df.loc[best_test_idx, 'C']:.4f} "
      f"(test_f1={results_df.loc[best_test_idx, 'test_f1']:.3f}, "
      f"test_acc={results_df.loc[best_test_idx, 'test_acc']:.3f}, "
      f"cv_f1={results_df.loc[best_test_idx, 'cv_f1_mean']:.3f}, "
      f"features={results_df.loc[best_test_idx, 'non_zero']})")

# Best balance (smallest gap with good CV F1)
# Define a score that balances CV F1 and its stability
results_df['balance_score'] = results_df['cv_f1_mean'] - results_df['cv_f1_std']
best_balance_idx = results_df['balance_score'].idxmax()
print(f"Best stable CV F1: C={results_df.loc[best_balance_idx, 'C']:.4f} "
      f"(cv_f1={results_df.loc[best_balance_idx, 'cv_f1_mean']:.3f}±{results_df.loc[best_balance_idx, 'cv_f1_std']:.3f}, "
      f"test_acc={results_df.loc[best_balance_idx, 'test_acc']:.3f}, "
      f"test_f1={results_df.loc[best_balance_idx, 'test_f1']:.3f}, "
      f"features={results_df.loc[best_balance_idx, 'non_zero']})")

# Most sparse with reasonable CV F1 (cv_f1_mean > 0.6)
reasonable_results = results_df[results_df['cv_f1_mean'] > 0.6]
if len(reasonable_results) > 0:
    sparsest_idx = reasonable_results['non_zero'].idxmin()
    print(f"Sparsest reasonable: C={reasonable_results.loc[sparsest_idx, 'C']:.4f} "
          f"(cv_f1={reasonable_results.loc[sparsest_idx, 'cv_f1_mean']:.3f}, "
          f"test_acc={reasonable_results.loc[sparsest_idx, 'test_acc']:.3f}, "
          f"test_f1={reasonable_results.loc[sparsest_idx, 'test_f1']:.3f}, "
          f"features={reasonable_results.loc[sparsest_idx, 'non_zero']})")

# %%
# Train final model with optimal C and analyze selected features
# Use best CV F1 for more robust selection
optimal_C = results_df.loc[best_cv_idx, 'C']
print(f"\nTraining final model with C={optimal_C:.4f} (best CV F1)")

clf_final = LogisticRegression(C=optimal_C, max_iter=2000, random_state=42, penalty='l1', solver='saga')
clf_final.fit(X_train_scaled, y_train)

# Get non-zero features
coefficients = clf_final.coef_[0]
non_zero_mask = np.abs(coefficients) > 1e-6
non_zero_indices = np.where(non_zero_mask)[0]
non_zero_coefs = coefficients[non_zero_mask]

print(f"\nSelected {len(non_zero_indices)} features")

# %%
# Analyze which layers and projections were selected
n_layers = 64  # From the original model
proj_types = ['gate_proj', 'up_proj', 'down_proj']

# Create a visualization of selected features
selected_features = np.zeros((3, n_layers))
coef_values = np.zeros((3, n_layers))

for idx, coef in zip(non_zero_indices, non_zero_coefs):
    proj_idx = idx // n_layers
    layer_idx = idx % n_layers
    selected_features[proj_idx, layer_idx] = 1
    coef_values[proj_idx, layer_idx] = coef

# Plot heatmap of selected features
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))

# Binary selection heatmap
sns.heatmap(selected_features, 
            xticklabels=range(n_layers),
            yticklabels=proj_types,
            cmap='Blues',
            cbar_kws={'label': 'Selected'},
            ax=ax1)
ax1.set_title(f'Selected Features by L1 Regularization (C={optimal_C:.4f})')
ax1.set_xlabel('Layer')

# Coefficient values heatmap
sns.heatmap(coef_values,
            xticklabels=range(n_layers),
            yticklabels=proj_types,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Coefficient'},
            ax=ax2)
ax2.set_title('Coefficient Values for Selected Features')
ax2.set_xlabel('Layer')

plt.tight_layout()
plt.show()

# %%
# Summary statistics by projection type
print("\nSummary by projection type:")
for proj_idx, proj_type in enumerate(proj_types):
    proj_start = proj_idx * n_layers
    proj_end = (proj_idx + 1) * n_layers
    
    selected_in_proj = [idx for idx in non_zero_indices if proj_start <= idx < proj_end]
    selected_layers = [idx - proj_start for idx in selected_in_proj]
    
    if selected_layers:
        coefs_in_proj = [coefficients[idx] for idx in selected_in_proj]
        print(f"\n{proj_type}:")
        print(f"  - Selected layers: {selected_layers}")
        print(f"  - Number selected: {len(selected_layers)}/{n_layers}")
        print(f"  - Mean |coefficient|: {np.mean(np.abs(coefs_in_proj)):.3f}")
        print(f"  - Max |coefficient|: {np.max(np.abs(coefs_in_proj)):.3f}")

# %%
# Cross-validation to ensure robustness
from sklearn.model_selection import cross_val_score, cross_validate

print("\n5-fold cross-validation with optimal C:")
cv_results = cross_validate(
    LogisticRegression(C=optimal_C, max_iter=2000, penalty='l1', solver='saga'),
    X_train_scaled, y_train, cv=5, scoring=['f1', 'accuracy']
)
print(f"CV F1 scores: {cv_results['test_f1']}")
print(f"Mean CV F1: {np.mean(cv_results['test_f1']):.3f} (+/- {np.std(cv_results['test_f1'])*2:.3f})")
print(f"Mean CV accuracy: {np.mean(cv_results['test_accuracy']):.3f} (+/- {np.std(cv_results['test_accuracy'])*2:.3f})")

# %%
print("\nExperiment complete!")
print(f"Best C value: {optimal_C:.4f}")
print(f"CV F1 score: {results_df.loc[best_cv_idx, 'cv_f1_mean']:.3f}±{results_df.loc[best_cv_idx, 'cv_f1_std']:.3f}")
print(f"Test F1 score: {results_df.loc[best_cv_idx, 'test_f1']:.3f}")
print(f"Test accuracy: {results_df.loc[best_cv_idx, 'test_acc']:.3f}")
print(f"Number of selected features: {results_df.loc[best_cv_idx, 'non_zero']}/192")
# %%
