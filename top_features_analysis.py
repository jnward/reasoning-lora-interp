# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
print(f"Class balance: {np.sum(y)/len(y)*100:.1f}% Yes, {(1-np.sum(y)/len(y))*100:.1f}% No")

# %%
# Create feature names
n_layers = 64
proj_types = ['gate_proj', 'up_proj', 'down_proj']
feature_names = []
for proj_type in proj_types:
    for layer in range(n_layers):
        feature_names.append(f'{proj_type}_L{layer}')

# %%
# Split and scale data
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, valid_indices, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} examples")
print(f"Test set: {len(X_test)} examples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Find top 3 features using multiple methods
print("\nFinding top 3 features using different methods...")

# Method 1: F-statistics
selector_f = SelectKBest(f_classif, k=3)
selector_f.fit(X_train_scaled, y_train)
scores_f = selector_f.scores_
top3_f_idx = np.argsort(scores_f)[-3:][::-1]

print("\nTop 3 features by F-statistic:")
for i, idx in enumerate(top3_f_idx):
    print(f"{i+1}. {feature_names[idx]} (F-score: {scores_f[idx]:.2f})")

# Method 2: Mutual information
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
top3_mi_idx = np.argsort(mi_scores)[-3:][::-1]

print("\nTop 3 features by Mutual Information:")
for i, idx in enumerate(top3_mi_idx):
    print(f"{i+1}. {feature_names[idx]} (MI-score: {mi_scores[idx]:.4f})")

# Method 3: L1 regularization coefficients (from previous analysis)
# Train L1 model with C=0.07
clf_l1 = LogisticRegression(C=0.07, max_iter=2000, random_state=42, penalty='l1', solver='saga')
clf_l1.fit(X_train_scaled, y_train)
coef_abs = np.abs(clf_l1.coef_[0])
top3_l1_idx = np.argsort(coef_abs)[-3:][::-1]

print("\nTop 3 features by L1 coefficient magnitude:")
for i, idx in enumerate(top3_l1_idx):
    print(f"{i+1}. {feature_names[idx]} (|coef|: {coef_abs[idx]:.4f})")

# %%
# Use F-statistic features for main analysis
top3_idx = top3_f_idx
print(f"\nUsing F-statistic top 3 features: {[feature_names[i] for i in top3_idx]}")

# Extract top 3 features
X_train_top3 = X_train_scaled[:, top3_idx]
X_test_top3 = X_test_scaled[:, top3_idx]

# %%
# Create interactive 3D scatter plot
print("\nCreating interactive 3D visualization...")

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('All Data with Top 3 Features', 'Train vs Test Data'),
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    horizontal_spacing=0.05
)

# Plot 1: All data colored by correctness
fig.add_trace(
    go.Scatter3d(
        x=X_train_top3[y_train==0, 0],
        y=X_train_top3[y_train==0, 1],
        z=X_train_top3[y_train==0, 2],
        mode='markers',
        marker=dict(color='red', size=5, opacity=0.6),
        name='Incorrect',
        text=[f'Sample {i}' for i in np.where(y_train==0)[0]],
        hovertemplate='%{text}<br>F1: %{x:.2f}<br>F2: %{y:.2f}<br>F3: %{z:.2f}<extra></extra>'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter3d(
        x=X_train_top3[y_train==1, 0],
        y=X_train_top3[y_train==1, 1],
        z=X_train_top3[y_train==1, 2],
        mode='markers',
        marker=dict(color='green', size=5, opacity=0.6),
        name='Correct',
        text=[f'Sample {i}' for i in np.where(y_train==1)[0]],
        hovertemplate='%{text}<br>F1: %{x:.2f}<br>F2: %{y:.2f}<br>F3: %{z:.2f}<extra></extra>'
    ),
    row=1, col=1
)

# Plot 2: Train vs Test
fig.add_trace(
    go.Scatter3d(
        x=X_train_top3[y_train==0, 0],
        y=X_train_top3[y_train==0, 1],
        z=X_train_top3[y_train==0, 2],
        mode='markers',
        marker=dict(color='red', size=4, opacity=0.5, symbol='circle'),
        name='Train Incorrect'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=X_train_top3[y_train==1, 0],
        y=X_train_top3[y_train==1, 1],
        z=X_train_top3[y_train==1, 2],
        mode='markers',
        marker=dict(color='green', size=4, opacity=0.5, symbol='circle'),
        name='Train Correct'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=X_test_top3[y_test==0, 0],
        y=X_test_top3[y_test==0, 1],
        z=X_test_top3[y_test==0, 2],
        mode='markers',
        marker=dict(color='darkred', size=8, opacity=0.9, symbol='diamond'),
        name='Test Incorrect'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=X_test_top3[y_test==1, 0],
        y=X_test_top3[y_test==1, 1],
        z=X_test_top3[y_test==1, 2],
        mode='markers',
        marker=dict(color='darkgreen', size=8, opacity=0.9, symbol='diamond'),
        name='Test Correct'
    ),
    row=1, col=2
)

# Update layout
fig.update_scenes(
    xaxis_title=feature_names[top3_idx[0]],
    yaxis_title=feature_names[top3_idx[1]],
    zaxis_title=feature_names[top3_idx[2]]
)

fig.update_layout(
    title='Interactive 3D Visualization with Top 3 Features',
    width=1400,
    height=700
)

fig.show()

# %%
# Train classifier on top 3 features
print("\nTraining logistic regression on top 3 features...")
clf_top3 = LogisticRegression(max_iter=1000, random_state=42)
clf_top3.fit(X_train_top3, y_train)

# Evaluate
train_acc = clf_top3.score(X_train_top3, y_train)
test_acc = clf_top3.score(X_test_top3, y_test)

y_train_pred = clf_top3.predict(X_train_top3)
y_test_pred = clf_top3.predict(X_test_top3)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\nPerformance with top 3 features:")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Train F1 score: {train_f1:.4f}")
print(f"Test F1 score: {test_f1:.4f}")

# 5-fold cross-validation
print("\n5-fold Cross-Validation (top 3 features):")
cv_results = cross_validate(
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_top3, y_train, cv=5, 
    scoring=['accuracy', 'f1']
)
print(f"CV accuracy: {np.mean(cv_results['test_accuracy']):.3f} (+/- {np.std(cv_results['test_accuracy'])*2:.3f})")
print(f"CV F1 score: {np.mean(cv_results['test_f1']):.3f} (+/- {np.std(cv_results['test_f1'])*2:.3f})")

# %%
# Create 2D projections
print("\nCreating 2D projections...")

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(f'{feature_names[top3_idx[0]]} vs {feature_names[top3_idx[1]]}',
                    f'{feature_names[top3_idx[0]]} vs {feature_names[top3_idx[2]]}',
                    f'{feature_names[top3_idx[1]]} vs {feature_names[top3_idx[2]]}'),
    horizontal_spacing=0.05
)

# Define projections
projections = [(0, 1), (0, 2), (1, 2)]

for col, (idx1, idx2) in enumerate(projections, 1):
    # Plot training data
    fig.add_trace(
        go.Scatter(
            x=X_train_top3[y_train==0, idx1],
            y=X_train_top3[y_train==0, idx2],
            mode='markers',
            marker=dict(color='red', size=8, opacity=0.6, line=dict(color='darkred', width=1)),
            name='Incorrect' if col == 1 else None,
            showlegend=(col == 1)
        ),
        row=1, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=X_train_top3[y_train==1, idx1],
            y=X_train_top3[y_train==1, idx2],
            mode='markers',
            marker=dict(color='green', size=8, opacity=0.6, line=dict(color='darkgreen', width=1)),
            name='Correct' if col == 1 else None,
            showlegend=(col == 1)
        ),
        row=1, col=col
    )

# Update axes
fig.update_xaxes(title_text=feature_names[top3_idx[0]], row=1, col=1)
fig.update_yaxes(title_text=feature_names[top3_idx[1]], row=1, col=1)
fig.update_xaxes(title_text=feature_names[top3_idx[0]], row=1, col=2)
fig.update_yaxes(title_text=feature_names[top3_idx[2]], row=1, col=2)
fig.update_xaxes(title_text=feature_names[top3_idx[1]], row=1, col=3)
fig.update_yaxes(title_text=feature_names[top3_idx[2]], row=1, col=3)

fig.update_layout(
    title='2D Projections of Top 3 Features',
    height=500,
    width=1400
)

fig.show()

# %%
# Analyze individual feature performance
print("\nAnalyzing individual feature performance...")

individual_results = []

for i, idx in enumerate(top3_idx):
    # Extract single feature
    X_train_single = X_train_scaled[:, idx].reshape(-1, 1)
    X_test_single = X_test_scaled[:, idx].reshape(-1, 1)
    
    # Train classifier
    clf_single = LogisticRegression(max_iter=1000, random_state=42)
    clf_single.fit(X_train_single, y_train)
    
    # Evaluate
    test_acc = clf_single.score(X_test_single, y_test)
    y_test_pred_single = clf_single.predict(X_test_single)
    test_f1 = f1_score(y_test, y_test_pred_single)
    
    # Cross-validation
    cv_results_single = cross_validate(
        LogisticRegression(max_iter=1000, random_state=42),
        X_train_single, y_train, cv=5,
        scoring=['accuracy', 'f1']
    )
    
    individual_results.append({
        'feature': feature_names[idx],
        'test_acc': test_acc,
        'test_f1': test_f1,
        'cv_f1_mean': np.mean(cv_results_single['test_f1']),
        'cv_f1_std': np.std(cv_results_single['test_f1']),
        'coefficient': clf_single.coef_[0][0],
        'intercept': clf_single.intercept_[0]
    })
    
    print(f"\n{feature_names[idx]}:")
    print(f"  Test accuracy: {test_acc:.3f}")
    print(f"  Test F1: {test_f1:.3f}")
    print(f"  CV F1: {np.mean(cv_results_single['test_f1']):.3f}±{np.std(cv_results_single['test_f1']):.3f}")
    print(f"  Coefficient: {clf_single.coef_[0][0]:.4f}")

# %%
# Compare performance
print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

comparison_data = {
    'Method': ['All 192 features (L1, C=0.07)', 'Top 3 features'] + [f'Single: {r["feature"]}' for r in individual_results],
    'Test Accuracy': ['-', test_acc] + [r['test_acc'] for r in individual_results],
    'Test F1': ['-', test_f1] + [r['test_f1'] for r in individual_results],
    'CV F1': ['-', f"{np.mean(cv_results['test_f1']):.3f}±{np.std(cv_results['test_f1']):.3f}"] + 
             [f"{r['cv_f1_mean']:.3f}±{r['cv_f1_std']:.3f}" for r in individual_results]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# %%
# Visualize feature distributions
print("\nVisualizing feature distributions...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, idx in enumerate(top3_idx):
    ax = axes[i]
    
    # Extract feature values
    feature_vals_train = X_train[:, idx]
    
    # Plot distributions
    ax.hist(feature_vals_train[y_train==0], bins=30, alpha=0.5, label='Incorrect', color='red', density=True)
    ax.hist(feature_vals_train[y_train==1], bins=30, alpha=0.5, label='Correct', color='green', density=True)
    
    ax.set_xlabel(f'{feature_names[idx]} (raw values)')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {feature_names[idx]}')
    ax.legend()
    
    # Add statistics
    mean_incorrect = feature_vals_train[y_train==0].mean()
    mean_correct = feature_vals_train[y_train==1].mean()
    ax.axvline(mean_incorrect, color='darkred', linestyle='--', alpha=0.7, label=f'Mean (No): {mean_incorrect:.3f}')
    ax.axvline(mean_correct, color='darkgreen', linestyle='--', alpha=0.7, label=f'Mean (Yes): {mean_correct:.3f}')

plt.tight_layout()
plt.show()

# %%
# Create a correlation matrix for top 3 features
print("\nAnalyzing correlation between top 3 features...")

correlation_matrix = np.corrcoef(X_train_top3.T)
print("\nCorrelation matrix:")
for i in range(3):
    for j in range(3):
        if i <= j:
            print(f"{feature_names[top3_idx[i]]} vs {feature_names[top3_idx[j]]}: {correlation_matrix[i,j]:.3f}")

# %%
# Try all pairs of features
print("\n" + "="*80)
print("PAIRWISE FEATURE ANALYSIS")
print("="*80)

pairwise_results = []

for i in range(3):
    for j in range(i+1, 3):
        # Extract pair of features
        pair_indices = [top3_idx[i], top3_idx[j]]
        X_train_pair = X_train_scaled[:, pair_indices]
        X_test_pair = X_test_scaled[:, pair_indices]
        
        # Train classifier
        clf_pair = LogisticRegression(max_iter=1000, random_state=42)
        clf_pair.fit(X_train_pair, y_train)
        
        # Evaluate
        test_acc = clf_pair.score(X_test_pair, y_test)
        y_test_pred_pair = clf_pair.predict(X_test_pair)
        test_f1 = f1_score(y_test, y_test_pred_pair)
        
        # Cross-validation
        cv_results_pair = cross_validate(
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_pair, y_train, cv=5,
            scoring=['accuracy', 'f1']
        )
        
        result = {
            'features': f"{feature_names[top3_idx[i]]} + {feature_names[top3_idx[j]]}",
            'test_acc': test_acc,
            'test_f1': test_f1,
            'cv_f1_mean': np.mean(cv_results_pair['test_f1']),
            'cv_f1_std': np.std(cv_results_pair['test_f1'])
        }
        
        pairwise_results.append(result)
        print(f"\n{result['features']}:")
        print(f"  Test F1: {test_f1:.3f}")
        print(f"  CV F1: {result['cv_f1_mean']:.3f}±{result['cv_f1_std']:.3f}")

# %%
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Top 3 features selected by F-statistic:")
for i, idx in enumerate(top3_idx):
    print(f"  {i+1}. {feature_names[idx]}")

print(f"\nPerformance with top 3 features:")
print(f"  Test F1: {test_f1:.3f}")
print(f"  CV F1: {np.mean(cv_results['test_f1']):.3f}±{np.std(cv_results['test_f1']):.3f}")

best_single = max(individual_results, key=lambda x: x['test_f1'])
print(f"\nBest single feature: {best_single['feature']} (F1: {best_single['test_f1']:.3f})")

if pairwise_results:
    best_pair = max(pairwise_results, key=lambda x: x['test_f1'])
    print(f"Best pair: {best_pair['features']} (F1: {best_pair['test_f1']:.3f})")

# %%