import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('tech_stocks_synthetic_data.csv', index_col='date', parse_dates=True)

# Select numerical variables for analysis
numerical_cols = ['daily_returns_pct', 'trading_volume_millions', 
                  'volatility_index', 'market_cap_change_pct']
data = df[numerical_cols]

print("=" * 70)
print("VARIANCE/COVARIANCE ANALYSIS: TECH STOCKS DATASET")
print("=" * 70)
print(f"\nDataset shape: {data.shape}")
print(f"Variables analyzed: {', '.join(numerical_cols)}")

# ============================================================================
# STEP 1: CALCULATE COVARIANCE MATRIX
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: COVARIANCE MATRIX")
print("=" * 70)

# Calculate covariance matrix
cov_matrix = data.cov()
print("\nCovariance Matrix:")
print(cov_matrix.round(3))

# Calculate correlation matrix for comparison
corr_matrix = data.corr()
print("\nCorrelation Matrix (for reference):")
print(corr_matrix.round(3))

# ============================================================================
# STEP 2: EIGENVALUES AND EIGENVECTORS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: EIGENVALUES AND EIGENVECTORS")
print("=" * 70)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalues (descending order)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nEigenvalues (sorted by magnitude):")
for i, val in enumerate(eigenvalues):
    variance_explained = (val / np.sum(eigenvalues)) * 100
    print(f"  λ{i+1} = {val:.4f} ({variance_explained:.2f}% of total variance)")

print("\nEigenvectors (columns are eigenvectors):")
eigenvector_df = pd.DataFrame(
    eigenvectors,
    index=numerical_cols,
    columns=[f'PC{i+1}' for i in range(len(eigenvalues))]
)
print(eigenvector_df.round(4))

# ============================================================================
# STEP 3: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: CREATING VISUALIZATIONS")
print("=" * 70)

# Create a figure with subplots
fig = plt.figure(figsize=(20, 12))

# --- 3.1: Scatter Plot Matrix ---
print("\n3.1: Creating scatter plot matrix...")
from matplotlib.gridspec import GridSpec

# Create main grid for better organization
gs_main = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)

# Subplot for scatter matrix (top-left quadrant)
gs_scatter = GridSpec(4, 4, figure=fig, 
                     left=0.05, right=0.45, top=0.95, bottom=0.52,
                     wspace=0.15, hspace=0.15)

# Create scatter plots for each pair of variables
for i in range(len(numerical_cols)):
    for j in range(len(numerical_cols)):
        ax = fig.add_subplot(gs_scatter[i, j])
        
        if i == j:
            # Diagonal: show distribution
            ax.hist(data[numerical_cols[i]], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            if j == 0:
                ax.set_ylabel('Freq', fontsize=7)
        else:
            # Off-diagonal: scatter plots
            ax.scatter(data[numerical_cols[j]], data[numerical_cols[i]], 
                      alpha=0.6, s=15, color='darkblue')
            
            # Add regression line
            z = np.polyfit(data[numerical_cols[j]], data[numerical_cols[i]], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[numerical_cols[j]].min(), 
                                data[numerical_cols[j]].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=0.8)
        
        # Labels - simplified
        if i == len(numerical_cols) - 1:
            labels = ['Returns', 'Volume', 'Volatility', 'Mkt Cap']
            ax.set_xlabel(labels[j], fontsize=7)
        if j == 0 and i != j:
            labels = ['Returns', 'Volume', 'Volatility', 'Mkt Cap']
            ax.set_ylabel(labels[i], fontsize=7)
        
        ax.tick_params(labelsize=5)

# Title for scatter matrix
fig.text(0.25, 0.98, 'Scatter Plot Matrix', ha='center', fontsize=12, fontweight='bold')

# --- 3.2: Covariance Matrix Heatmap (top-right) ---
print("3.2: Creating covariance matrix heatmap...")
ax_cov = fig.add_subplot(gs_main[0, 1])
im = ax_cov.imshow(cov_matrix, cmap='RdBu_r', aspect='auto')
ax_cov.set_xticks(range(len(numerical_cols)))
ax_cov.set_yticks(range(len(numerical_cols)))

# Shorter labels for readability
short_labels = ['Returns\n(%)', 'Volume\n(M)', 'Volatility', 'Mkt Cap\n(%)']
ax_cov.set_xticklabels(short_labels, fontsize=9)
ax_cov.set_yticklabels(short_labels, fontsize=9)

# Add values to heatmap
for i in range(len(numerical_cols)):
    for j in range(len(numerical_cols)):
        value = cov_matrix.iloc[i, j]
        text_color = "white" if abs(value) > 150 else "black"
        ax_cov.text(j, i, f'{value:.1f}',
                   ha="center", va="center", color=text_color, fontsize=8)

ax_cov.set_title('Covariance Matrix Heatmap', fontsize=11, pad=10)
plt.colorbar(im, ax=ax_cov, fraction=0.046, pad=0.04)

# --- 3.3: Principal Component Analysis Visualization (bottom-left) ---
print("3.3: Creating PCA visualization...")

# Standardize the data for PCA
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(data_standardized)

# Create subplot for PCA and Biplot side by side
gs_bottom = GridSpec(1, 3, figure=fig, 
                    left=0.05, right=0.95, top=0.42, bottom=0.05,
                    wspace=0.3)

# PCA scatter plot
ax_pca = fig.add_subplot(gs_bottom[0, 0])
scatter = ax_pca.scatter(principal_components[:, 0], principal_components[:, 1], 
                         c=df.index.astype(np.int64) // 10**9, # Color by date
                         cmap='viridis', alpha=0.6, s=30)
ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=9)
ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=9)
ax_pca.set_title('First Two Principal Components', fontsize=10)
ax_pca.grid(True, alpha=0.3)

# --- 3.4: Scree Plot (bottom-middle) ---
print("3.4: Creating scree plot...")
ax_scree = fig.add_subplot(gs_bottom[0, 1])
variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(variance_ratio)

bars = ax_scree.bar(range(1, len(eigenvalues) + 1), variance_ratio * 100, 
                   alpha=0.7, color='steelblue', label='Individual')
line = ax_scree.plot(range(1, len(eigenvalues) + 1), cumulative_variance * 100, 
                    'ro-', linewidth=2, markersize=6, label='Cumulative')

ax_scree.set_xlabel('Principal Component', fontsize=9)
ax_scree.set_ylabel('Variance Explained (%)', fontsize=9)
ax_scree.set_title('Scree Plot', fontsize=10)
ax_scree.set_xticks(range(1, len(eigenvalues) + 1))
ax_scree.grid(True, alpha=0.3)
ax_scree.legend(fontsize=8, loc='center right')

# Add percentage labels
for i, val in enumerate(variance_ratio * 100):
    if val > 1:  # Only show labels for significant components
        ax_scree.text(i + 1, val + 1, f'{val:.1f}%', ha='center', fontsize=7)

# --- 3.5: Biplot (bottom-right) ---
print("3.5: Creating biplot...")
ax_biplot = fig.add_subplot(gs_bottom[0, 2])

# Plot data points in PC space
ax_biplot.scatter(principal_components[:, 0], principal_components[:, 1], 
                  alpha=0.2, s=15, color='gray')

# Plot variable loadings as arrows
feature_vectors = pca.components_[:2].T
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # Distinct colors for each variable
short_names = ['Returns', 'Volume', 'Volatility', 'MktCap']

for i, (feature, vec, color, name) in enumerate(zip(numerical_cols, feature_vectors, colors, short_names)):
    # Scale arrows for visibility
    scale = 3
    ax_biplot.arrow(0, 0, vec[0] * scale, vec[1] * scale, 
                   color=color, alpha=0.8, linewidth=2,
                   head_width=0.12, head_length=0.08)
    
    # Add labels with background for readability
    label_pos = vec * scale * 1.2
    ax_biplot.text(label_pos[0], label_pos[1], name, 
                  fontsize=9, ha='center', fontweight='bold',
                  color=color, bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='white', alpha=0.7))

ax_biplot.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
ax_biplot.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
ax_biplot.set_title('Biplot - Variables in PC Space', fontsize=10)
ax_biplot.grid(True, alpha=0.3)
ax_biplot.axhline(y=0, color='k', linewidth=0.5)
ax_biplot.axvline(x=0, color='k', linewidth=0.5)
ax_biplot.set_xlim(-4, 4)
ax_biplot.set_ylim(-4, 4)

plt.suptitle('Variance/Covariance Analysis - Tech Stocks Dataset', fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.show()

# ============================================================================
# STEP 4: INTERPRET THE LARGEST EIGENVECTOR
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: INTERPRETATION OF THE LARGEST EIGENVECTOR")
print("=" * 70)

# Get the first eigenvector (corresponding to largest eigenvalue)
first_eigenvector = eigenvectors[:, 0]
print(f"\nLargest eigenvalue: {eigenvalues[0]:.4f}")
print(f"Variance explained: {(eigenvalues[0]/np.sum(eigenvalues))*100:.2f}%")

print("\nFirst Principal Component Loadings:")
for var, loading in zip(numerical_cols, first_eigenvector):
    print(f"  {var:30s}: {loading:+.4f}")

# Find the dominant variables
abs_loadings = np.abs(first_eigenvector)
dominant_idx = np.argsort(abs_loadings)[-2:]  # Top 2 contributors
dominant_vars = [numerical_cols[i] for i in dominant_idx]

print("\n" + "=" * 70)
