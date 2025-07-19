# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
import pandas as pd

# %%
# Load dataset and filter by solution length
print("Loading dataset and filtering for short solutions...")
dataset = load_dataset("simplescaling/s1K-1.1", split="train")

# Filter for short solutions
short_solution_indices = []
for i, example in enumerate(dataset):
    solution = example.get('solution', '')
    if len(solution) <= 64:
        short_solution_indices.append(i)

# %%
# Load cached features and filter
cache_data = np.load('lora_activation_cache/summary_activations_n1000_last16_r1.npz')
X_all = cache_data['X']
y_all = cache_data['y']
valid_indices_all = cache_data['valid_indices']

# Find intersection of valid indices and short solution indices
valid_short_indices = [i for i in range(len(valid_indices_all)) 
                      if valid_indices_all[i] in short_solution_indices]

X = X_all[valid_short_indices]
y = y_all[valid_short_indices]

print(f"Total examples after filtering: {len(X)}")
print(f"Class distribution: {np.sum(y)} Yes, {len(y) - np.sum(y)} No")

# %%
# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Train class-weighted L1 with C=0.1 to identify top features
clf = LogisticRegression(C=0.1, max_iter=2000, random_state=42, 
                        penalty='l1', solver='saga', class_weight='balanced')
clf.fit(X_train_scaled, y_train)

# Get feature importance
coefficients = clf.coef_[0]
feature_importance = np.abs(coefficients)
top2_indices = np.argsort(feature_importance)[-2:][::-1]

# Create feature names
n_layers = 64
proj_types = ['gate_proj', 'up_proj', 'down_proj']
feature_names = []
for proj_type in proj_types:
    for layer in range(n_layers):
        feature_names.append(f'{proj_type}_L{layer}')

print(f"\nTop 2 features:")
print(f"1. {feature_names[top2_indices[0]]} (coef={coefficients[top2_indices[0]]:.4f})")
print(f"2. {feature_names[top2_indices[1]]} (coef={coefficients[top2_indices[1]]:.4f})")

# %%
# Create scatter plot with original (unscaled) features
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Training data
ax = axes[0]
# Extract top 2 features from training data
X_train_top2 = X_train[:, top2_indices]

# Separate by class
train_yes = X_train_top2[y_train == 1]
train_no = X_train_top2[y_train == 0]

ax.scatter(train_yes[:, 0], train_yes[:, 1], c='green', alpha=0.6, 
           label=f'Correct (n={len(train_yes)})', s=50)
ax.scatter(train_no[:, 0], train_no[:, 1], c='red', alpha=0.8, 
           label=f'Incorrect (n={len(train_no)})', s=80, marker='^')

ax.set_xlabel(feature_names[top2_indices[0]])
ax.set_ylabel(feature_names[top2_indices[1]])
ax.set_title('Training Data - Top 2 Features')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Test data
ax = axes[1]
X_test_top2 = X_test[:, top2_indices]

test_yes = X_test_top2[y_test == 1]
test_no = X_test_top2[y_test == 0]

ax.scatter(test_yes[:, 0], test_yes[:, 1], c='green', alpha=0.6, 
           label=f'Correct (n={len(test_yes)})', s=50)
ax.scatter(test_no[:, 0], test_no[:, 1], c='red', alpha=0.8, 
           label=f'Incorrect (n={len(test_no)})', s=80, marker='^')

ax.set_xlabel(feature_names[top2_indices[0]])
ax.set_ylabel(feature_names[top2_indices[1]])
ax.set_title('Test Data - Top 2 Features')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Create a more detailed plot with scaled features and decision boundary
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Get scaled top 2 features
X_train_top2_scaled = X_train_scaled[:, top2_indices]
X_test_top2_scaled = X_test_scaled[:, top2_indices]

# Train a 2-feature classifier
clf_2d = LogisticRegression(class_weight='balanced', random_state=42)
clf_2d.fit(X_train_top2_scaled, y_train)

# Create a mesh for decision boundary
h = .02  # step size in the mesh
x_min, x_max = X_train_top2_scaled[:, 0].min() - 1, X_train_top2_scaled[:, 0].max() + 1
y_min, y_max = X_train_top2_scaled[:, 1].min() - 1, X_train_top2_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh
Z = clf_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot decision boundary
contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), 
                      cmap='RdYlGn', alpha=0.4)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

# Plot points
train_yes_scaled = X_train_top2_scaled[y_train == 1]
train_no_scaled = X_train_top2_scaled[y_train == 0]

ax.scatter(train_yes_scaled[:, 0], train_yes_scaled[:, 1], 
           c='green', edgecolor='darkgreen', alpha=0.7, 
           label=f'Train Correct (n={len(train_yes_scaled)})', s=60)
ax.scatter(train_no_scaled[:, 0], train_no_scaled[:, 1], 
           c='red', edgecolor='darkred', alpha=0.9, 
           label=f'Train Incorrect (n={len(train_no_scaled)})', s=100, marker='^')

# Add test points with different markers
test_yes_scaled = X_test_top2_scaled[y_test == 1]
test_no_scaled = X_test_top2_scaled[y_test == 0]

ax.scatter(test_yes_scaled[:, 0], test_yes_scaled[:, 1], 
           c='lightgreen', edgecolor='darkgreen', alpha=0.9, 
           label=f'Test Correct (n={len(test_yes_scaled)})', s=60, marker='s')
ax.scatter(test_no_scaled[:, 0], test_no_scaled[:, 1], 
           c='pink', edgecolor='darkred', alpha=0.9, 
           label=f'Test Incorrect (n={len(test_no_scaled)})', s=100, marker='D')

ax.set_xlabel(f'{feature_names[top2_indices[0]]} (scaled)')
ax.set_ylabel(f'{feature_names[top2_indices[1]]} (scaled)')
ax.set_title('Top 2 Features with Decision Boundary (Scaled Features)')
ax.legend()
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('P(Correct)', rotation=270, labelpad=15)

# Print performance
y_pred_train = clf_2d.predict(X_train_top2_scaled)
y_pred_test = clf_2d.predict(X_test_top2_scaled)
from sklearn.metrics import f1_score
train_f1 = f1_score(y_train, y_pred_train)
test_f1 = f1_score(y_test, y_pred_test)

plt.text(0.02, 0.02, f'2-feature classifier:\nTrain F1: {train_f1:.3f}\nTest F1: {test_f1:.3f}', 
         transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# %%
# Distribution plots for each feature
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for i, ax in enumerate(axes):
    feature_idx = top2_indices[i]
    feature_name = feature_names[feature_idx]
    
    # Get feature values
    train_vals_yes = X_train[y_train == 1, feature_idx]
    train_vals_no = X_train[y_train == 0, feature_idx]
    
    # Plot distributions
    ax.hist(train_vals_yes, bins=30, alpha=0.5, label='Correct', color='green', density=True)
    ax.hist(train_vals_no, bins=30, alpha=0.5, label='Incorrect', color='red', density=True)
    
    ax.axvline(train_vals_yes.mean(), color='darkgreen', linestyle='--', 
               label=f'Mean (Yes): {train_vals_yes.mean():.3f}')
    ax.axvline(train_vals_no.mean(), color='darkred', linestyle='--', 
               label=f'Mean (No): {train_vals_no.mean():.3f}')
    
    ax.set_xlabel(f'{feature_name} activation')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {feature_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add coefficient info
    coef = coefficients[feature_idx]
    ax.text(0.02, 0.98, f'L1 coefficient: {coef:.4f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# %%
print("\nSummary:")
print(f"Using only the top 2 features achieves Test F1: {test_f1:.3f}")
print(f"This is {test_f1/0.908*100:.1f}% of the full 22-feature model's performance (F1=0.908)")

# %%