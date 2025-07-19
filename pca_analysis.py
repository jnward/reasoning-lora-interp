# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
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
print(f"Class balance: {np.sum(y)/len(y)*100:.1f}% Yes, {(1-np.sum(y)/len(y))*100:.1f}% No")

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
# Apply PCA to get top 2 components
print("\nApplying PCA to extract top 2 components...")
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained by 2 components: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")

# %%
# Create interactive 2D scatter plots with Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplots
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Training Data in PCA Space', 
                    'Test Data in PCA Space', 
                    'Train vs Test Data in PCA Space'),
    horizontal_spacing=0.05
)

# Plot 1: Training data
fig.add_trace(
    go.Scatter(
        x=X_train_pca[y_train==0, 0],
        y=X_train_pca[y_train==0, 1],
        mode='markers',
        marker=dict(color='red', size=8, opacity=0.6, line=dict(color='darkred', width=1)),
        name='Train Incorrect',
        showlegend=False
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=X_train_pca[y_train==1, 0],
        y=X_train_pca[y_train==1, 1],
        mode='markers',
        marker=dict(color='green', size=8, opacity=0.6, line=dict(color='darkgreen', width=1)),
        name='Train Correct',
        showlegend=False
    ),
    row=1, col=1
)

# Plot 2: Test data
fig.add_trace(
    go.Scatter(
        x=X_test_pca[y_test==0, 0],
        y=X_test_pca[y_test==0, 1],
        mode='markers',
        marker=dict(color='red', size=8, opacity=0.6, line=dict(color='darkred', width=1)),
        name='Test Incorrect',
        showlegend=False
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=X_test_pca[y_test==1, 0],
        y=X_test_pca[y_test==1, 1],
        mode='markers',
        marker=dict(color='green', size=8, opacity=0.6, line=dict(color='darkgreen', width=1)),
        name='Test Correct',
        showlegend=False
    ),
    row=1, col=2
)

# Plot 3: Both sets with different markers
fig.add_trace(
    go.Scatter(
        x=X_train_pca[y_train==0, 0],
        y=X_train_pca[y_train==0, 1],
        mode='markers',
        marker=dict(color='red', size=6, opacity=0.5),
        name='Train Incorrect'
    ),
    row=1, col=3
)
fig.add_trace(
    go.Scatter(
        x=X_train_pca[y_train==1, 0],
        y=X_train_pca[y_train==1, 1],
        mode='markers',
        marker=dict(color='green', size=6, opacity=0.5),
        name='Train Correct'
    ),
    row=1, col=3
)
fig.add_trace(
    go.Scatter(
        x=X_test_pca[y_test==0, 0],
        y=X_test_pca[y_test==0, 1],
        mode='markers',
        marker=dict(color='darkred', size=10, opacity=0.8, symbol='diamond'),
        name='Test Incorrect'
    ),
    row=1, col=3
)
fig.add_trace(
    go.Scatter(
        x=X_test_pca[y_test==1, 0],
        y=X_test_pca[y_test==1, 1],
        mode='markers',
        marker=dict(color='darkgreen', size=10, opacity=0.8, symbol='diamond'),
        name='Test Correct'
    ),
    row=1, col=3
)

# Update axes labels
for col in range(1, 4):
    fig.update_xaxes(title_text=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', row=1, col=col)
    fig.update_yaxes(title_text=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', row=1, col=col)

# Update layout
fig.update_layout(
    title_text="Interactive 2D PCA Visualization",
    height=500,
    width=1400,
    showlegend=True
)

fig.show()

# %%
# Train classifier on 2 PCA components
print("\nTraining logistic regression on 2 PCA components...")
clf_pca = LogisticRegression(max_iter=1000, random_state=42)
clf_pca.fit(X_train_pca, y_train)

# Evaluate
train_acc = clf_pca.score(X_train_pca, y_train)
test_acc = clf_pca.score(X_test_pca, y_test)

y_train_pred = clf_pca.predict(X_train_pca)
y_test_pred = clf_pca.predict(X_test_pca)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\nPerformance with 2 PCA components:")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Train F1 score: {train_f1:.4f}")
print(f"Test F1 score: {test_f1:.4f}")

# 5-fold cross-validation
print("\n5-fold Cross-Validation (2 PCA components):")
cv_results = cross_validate(
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_pca, y_train, cv=5, 
    scoring=['accuracy', 'f1']
)
print(f"CV accuracy: {np.mean(cv_results['test_accuracy']):.3f} (+/- {np.std(cv_results['test_accuracy'])*2:.3f})")
print(f"CV F1 score: {np.mean(cv_results['test_f1']):.3f} (+/- {np.std(cv_results['test_f1'])*2:.3f})")

# %%
# Visualize decision boundary with Plotly
print("\nVisualizing decision boundary...")

# Create a mesh
h = .02  # step size in the mesh
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh
Z = clf_pca.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Create interactive plot with Plotly
fig = go.Figure()

# Add contour plot for decision boundary
fig.add_trace(go.Contour(
    x=xx[0, :],
    y=yy[:, 0],
    z=Z,
    colorscale='RdYlGn',
    opacity=0.4,
    contours=dict(
        start=0,
        end=1,
        size=0.1,
    ),
    colorbar=dict(title='P(Correct)')
))

# Add decision boundary line
fig.add_trace(go.Contour(
    x=xx[0, :],
    y=yy[:, 0],
    z=Z,
    contours=dict(
        start=0.5,
        end=0.5,
        size=0,
        coloring='lines',
    ),
    line=dict(color='black', width=3, dash='dash'),
    showscale=False,
    name='Decision Boundary'
))

# Add training points
fig.add_trace(go.Scatter(
    x=X_train_pca[y_train==0, 0],
    y=X_train_pca[y_train==0, 1],
    mode='markers',
    marker=dict(color='red', size=8, line=dict(color='darkred', width=1)),
    name='Incorrect'
))

fig.add_trace(go.Scatter(
    x=X_train_pca[y_train==1, 0],
    y=X_train_pca[y_train==1, 1],
    mode='markers',
    marker=dict(color='green', size=8, line=dict(color='darkgreen', width=1)),
    name='Correct'
))

fig.update_layout(
    title='Interactive Decision Boundary in PCA Space',
    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)',
    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)',
    width=800,
    height=700
)

fig.show()

# %%
# Compare with more PCA components
print("\nComparing performance with different numbers of PCA components...")

n_components_list = [2, 5, 10, 20, 30, 50, 100, 192]
results = []

for n_comp in n_components_list:
    if n_comp > X_train_scaled.shape[1]:
        n_comp = X_train_scaled.shape[1]
    
    # Apply PCA
    pca_temp = PCA(n_components=n_comp, random_state=42)
    X_train_pca_temp = pca_temp.fit_transform(X_train_scaled)
    X_test_pca_temp = pca_temp.transform(X_test_scaled)
    
    # Train classifier
    clf_temp = LogisticRegression(max_iter=1000, random_state=42)
    clf_temp.fit(X_train_pca_temp, y_train)
    
    # Evaluate
    test_acc = clf_temp.score(X_test_pca_temp, y_test)
    y_test_pred = clf_temp.predict(X_test_pca_temp)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Cross-validation
    cv_results = cross_validate(
        LogisticRegression(max_iter=1000, random_state=42),
        X_train_pca_temp, y_train, cv=5, 
        scoring=['accuracy', 'f1']
    )
    
    var_explained = np.sum(pca_temp.explained_variance_ratio_) * 100
    
    results.append({
        'n_components': n_comp,
        'var_explained': var_explained,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'cv_acc_mean': np.mean(cv_results['test_accuracy']),
        'cv_acc_std': np.std(cv_results['test_accuracy']),
        'cv_f1_mean': np.mean(cv_results['test_f1']),
        'cv_f1_std': np.std(cv_results['test_f1'])
    })
    
    print(f"n_comp={n_comp}: Var={var_explained:.1f}%, "
          f"Test Acc={test_acc:.3f}, Test F1={test_f1:.3f}, "
          f"CV F1={results[-1]['cv_f1_mean']:.3f}±{results[-1]['cv_f1_std']:.3f}")

results_df = pd.DataFrame(results)

# %%
# Plot performance vs number of components
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Test metrics vs components
ax = axes[0, 0]
ax.plot(results_df['n_components'], results_df['test_acc'], 'o-', label='Test Accuracy', markersize=8)
ax.plot(results_df['n_components'], results_df['test_f1'], 's-', label='Test F1', markersize=8)
ax.set_xlabel('Number of PCA Components')
ax.set_ylabel('Score')
ax.set_title('Test Performance vs PCA Components')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 2: CV metrics vs components
ax = axes[0, 1]
ax.errorbar(results_df['n_components'], results_df['cv_acc_mean'], 
            yerr=results_df['cv_acc_std']*2, fmt='o-', label='CV Accuracy', markersize=8)
ax.errorbar(results_df['n_components'], results_df['cv_f1_mean'], 
            yerr=results_df['cv_f1_std']*2, fmt='s-', label='CV F1', markersize=8)
ax.set_xlabel('Number of PCA Components')
ax.set_ylabel('Score')
ax.set_title('Cross-Validation Performance vs PCA Components')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 3: Variance explained vs components
ax = axes[1, 0]
ax.plot(results_df['n_components'], results_df['var_explained'], 'g^-', markersize=8)
ax.set_xlabel('Number of PCA Components')
ax.set_ylabel('Cumulative Variance Explained (%)')
ax.set_title('Variance Explained vs PCA Components')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 4: Performance vs variance explained
ax = axes[1, 1]
ax.scatter(results_df['var_explained'], results_df['test_acc'], label='Test Accuracy', s=100, alpha=0.7)
ax.scatter(results_df['var_explained'], results_df['test_f1'], label='Test F1', s=100, alpha=0.7)
ax.scatter(results_df['var_explained'], results_df['cv_f1_mean'], label='CV F1', s=100, alpha=0.7)
ax.set_xlabel('Variance Explained (%)')
ax.set_ylabel('Score')
ax.set_title('Performance vs Variance Explained')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Analyze the top 2 principal components
print("\nAnalyzing the top 2 principal components...")

# Get the loadings (components)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create a DataFrame for easier analysis
feature_names = []
proj_types = ['gate_proj', 'up_proj', 'down_proj']
n_layers = 64

for proj_type in proj_types:
    for layer in range(n_layers):
        feature_names.append(f'{proj_type}_L{layer}')

loadings_df = pd.DataFrame(
    loadings,
    index=feature_names,
    columns=['PC1', 'PC2']
)

# Find top contributing features for each PC
print("\nTop 10 features contributing to PC1:")
pc1_top = loadings_df['PC1'].abs().nlargest(10)
for feat, val in pc1_top.items():
    print(f"  {feat}: {loadings_df.loc[feat, 'PC1']:.4f}")

print("\nTop 10 features contributing to PC2:")
pc2_top = loadings_df['PC2'].abs().nlargest(10)
for feat, val in pc2_top.items():
    print(f"  {feat}: {loadings_df.loc[feat, 'PC2']:.4f}")

# %%
# Create a biplot showing both data points and feature loadings
plt.figure(figsize=(12, 10))

# Plot data points
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                     c=y_train, cmap='RdYlGn', alpha=0.5, s=30)

# Plot top feature loadings as arrows
n_features_to_show = 20
top_features_idx = np.argsort(np.sum(loadings**2, axis=1))[-n_features_to_show:]

# Scale factor for arrows
scale = 3

for idx in top_features_idx:
    plt.arrow(0, 0, loadings[idx, 0]*scale, loadings[idx, 1]*scale,
              head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.7)
    plt.text(loadings[idx, 0]*scale*1.1, loadings[idx, 1]*scale*1.1, 
             feature_names[idx], fontsize=8, ha='center', va='center')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
plt.title('PCA Biplot: Data Points and Feature Loadings')
plt.colorbar(scatter, label='Answer (0=Incorrect, 1=Correct)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.show()

# %%
print("\nPCA Analysis Complete!")
print(f"With just 2 components explaining {np.sum(pca.explained_variance_ratio_)*100:.1f}% of variance:")
print(f"  - Test Accuracy: {test_acc:.3f}")
print(f"  - Test F1 Score: {test_f1:.3f}")
print(f"  - CV F1 Score: {np.mean(cv_results['test_f1']):.3f}±{np.std(cv_results['test_f1']):.3f}")

# %%
# 3D PCA Analysis
print("\n" + "="*80)
print("3D PCA ANALYSIS")
print("="*80)

# Apply PCA with 3 components
pca_3d = PCA(n_components=3, random_state=42)
X_train_pca_3d = pca_3d.fit_transform(X_train_scaled)
X_test_pca_3d = pca_3d.transform(X_test_scaled)

print(f"\nExplained variance ratio (3 components): {pca_3d.explained_variance_ratio_}")
print(f"Total variance explained by 3 components: {np.sum(pca_3d.explained_variance_ratio_)*100:.1f}%")

# %%
# Create interactive 3D scatter plots with Plotly
print("\nCreating interactive 3D visualizations...")

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('All Data in 3D PCA Space', 'Train vs Test in 3D PCA Space'),
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    horizontal_spacing=0.05
)

# Plot 1: All data colored by correctness
fig.add_trace(
    go.Scatter3d(
        x=X_train_pca_3d[y_train==0, 0],
        y=X_train_pca_3d[y_train==0, 1],
        z=X_train_pca_3d[y_train==0, 2],
        mode='markers',
        marker=dict(color='red', size=5, opacity=0.6),
        name='Incorrect',
        showlegend=True
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter3d(
        x=X_train_pca_3d[y_train==1, 0],
        y=X_train_pca_3d[y_train==1, 1],
        z=X_train_pca_3d[y_train==1, 2],
        mode='markers',
        marker=dict(color='green', size=5, opacity=0.6),
        name='Correct',
        showlegend=True
    ),
    row=1, col=1
)

# Plot 2: Train vs Test with different markers
fig.add_trace(
    go.Scatter3d(
        x=X_train_pca_3d[y_train==0, 0],
        y=X_train_pca_3d[y_train==0, 1],
        z=X_train_pca_3d[y_train==0, 2],
        mode='markers',
        marker=dict(color='red', size=4, opacity=0.5, symbol='circle'),
        name='Train Incorrect'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=X_train_pca_3d[y_train==1, 0],
        y=X_train_pca_3d[y_train==1, 1],
        z=X_train_pca_3d[y_train==1, 2],
        mode='markers',
        marker=dict(color='green', size=4, opacity=0.5, symbol='circle'),
        name='Train Correct'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=X_test_pca_3d[y_test==0, 0],
        y=X_test_pca_3d[y_test==0, 1],
        z=X_test_pca_3d[y_test==0, 2],
        mode='markers',
        marker=dict(color='darkred', size=8, opacity=0.9, symbol='diamond'),
        name='Test Incorrect'
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter3d(
        x=X_test_pca_3d[y_test==1, 0],
        y=X_test_pca_3d[y_test==1, 1],
        z=X_test_pca_3d[y_test==1, 2],
        mode='markers',
        marker=dict(color='darkgreen', size=8, opacity=0.9, symbol='diamond'),
        name='Test Correct'
    ),
    row=1, col=2
)

# Update layout for both subplots
fig.update_scenes(
    xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)',
    yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)',
    zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)'
)

fig.update_layout(
    title='Interactive 3D PCA Visualization',
    width=1400,
    height=700
)

fig.show()
print("Interactive 3D plots created! You can rotate, zoom, and pan to explore the data.")

# %%
# Train classifier on 3 PCA components
print("\nTraining logistic regression on 3 PCA components...")
clf_pca_3d = LogisticRegression(max_iter=1000, random_state=42)
clf_pca_3d.fit(X_train_pca_3d, y_train)

# Evaluate
train_acc_3d = clf_pca_3d.score(X_train_pca_3d, y_train)
test_acc_3d = clf_pca_3d.score(X_test_pca_3d, y_test)

y_train_pred_3d = clf_pca_3d.predict(X_train_pca_3d)
y_test_pred_3d = clf_pca_3d.predict(X_test_pca_3d)

train_f1_3d = f1_score(y_train, y_train_pred_3d)
test_f1_3d = f1_score(y_test, y_test_pred_3d)

print(f"\nPerformance with 3 PCA components:")
print(f"Train accuracy: {train_acc_3d:.4f}")
print(f"Test accuracy: {test_acc_3d:.4f}")
print(f"Train F1 score: {train_f1_3d:.4f}")
print(f"Test F1 score: {test_f1_3d:.4f}")

# 5-fold cross-validation
print("\n5-fold Cross-Validation (3 PCA components):")
cv_results_3d = cross_validate(
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_pca_3d, y_train, cv=5, 
    scoring=['accuracy', 'f1']
)
print(f"CV accuracy: {np.mean(cv_results_3d['test_accuracy']):.3f} (+/- {np.std(cv_results_3d['test_accuracy'])*2:.3f})")
print(f"CV F1 score: {np.mean(cv_results_3d['test_f1']):.3f} (+/- {np.std(cv_results_3d['test_f1'])*2:.3f})")


# %%
# Compare 2D vs 3D PCA performance
print("\n" + "="*80)
print("COMPARISON: 2D vs 3D PCA")
print("="*80)

comparison_data = {
    'Method': ['2 PCA Components', '3 PCA Components', 'All Features (192)'],
    'Variance Explained': [
        f"{np.sum(pca.explained_variance_ratio_)*100:.1f}%",
        f"{np.sum(pca_3d.explained_variance_ratio_)*100:.1f}%",
        "100.0%"
    ],
    'Test Accuracy': [test_acc, test_acc_3d, '-'],
    'Test F1': [test_f1, test_f1_3d, '-'],
    'CV F1': [
        f"{np.mean(cv_results['test_f1']):.3f}±{np.std(cv_results['test_f1']):.3f}",
        f"{np.mean(cv_results_3d['test_f1']):.3f}±{np.std(cv_results_3d['test_f1']):.3f}",
        '-'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# %%
# Analyze top 3 principal components
print("\n" + "="*80)
print("TOP 3 PRINCIPAL COMPONENTS ANALYSIS")
print("="*80)

# Get the loadings for 3 components
loadings_3d = pca_3d.components_.T * np.sqrt(pca_3d.explained_variance_)

loadings_3d_df = pd.DataFrame(
    loadings_3d,
    index=feature_names,
    columns=['PC1', 'PC2', 'PC3']
)

# Find top contributing features for PC3
print("\nTop 10 features contributing to PC3:")
pc3_top = loadings_3d_df['PC3'].abs().nlargest(10)
for feat, val in pc3_top.items():
    print(f"  {feat}: {loadings_3d_df.loc[feat, 'PC3']:.4f}")

# %%
# Create 2D projections of the 3D data
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

projections = [
    ('PC1', 'PC2', 0, 1),
    ('PC1', 'PC3', 0, 2),
    ('PC2', 'PC3', 1, 2)
]

for ax, (xlabel, ylabel, idx1, idx2) in zip(axes, projections):
    # Plot training data
    scatter = ax.scatter(X_train_pca_3d[:, idx1], X_train_pca_3d[:, idx2],
                        c=y_train, cmap='RdYlGn', alpha=0.6, s=50,
                        edgecolors='black', linewidth=0.5)
    ax.set_xlabel(f'{xlabel} ({pca_3d.explained_variance_ratio_[idx1]*100:.1f}%)')
    ax.set_ylabel(f'{ylabel} ({pca_3d.explained_variance_ratio_[idx2]*100:.1f}%)')
    ax.set_title(f'{xlabel} vs {ylabel} Projection')
    ax.grid(True, alpha=0.3)

plt.suptitle('2D Projections of 3D PCA Space', fontsize=16)
plt.tight_layout()
plt.show()

# %%
print("\n3D PCA Analysis Complete!")
print(f"Adding the 3rd component (explaining {pca_3d.explained_variance_ratio_[2]*100:.1f}% additional variance):")
print(f"  - Improved test accuracy from {test_acc:.3f} to {test_acc_3d:.3f}")
print(f"  - Improved test F1 from {test_f1:.3f} to {test_f1_3d:.3f}")
print(f"  - Total variance explained increased from {np.sum(pca.explained_variance_ratio_)*100:.1f}% to {np.sum(pca_3d.explained_variance_ratio_)*100:.1f}%")

# %%
