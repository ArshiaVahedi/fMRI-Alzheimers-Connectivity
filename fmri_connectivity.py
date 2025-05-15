import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.stats import zscore
import networkx as nx

# Define a simple HRF function
def hrf(t):
    return t**5 * np.exp(-t / 1.0)

# Simulation parameters
n_rois = 5
n_timepoints = 200
tr = 2.0  # Repetition time in seconds
roi_names = ['Motor', 'Sensory', 'Hippocampus', 'Prefrontal', 'Occipital']

# Generate HRF signal
t_hrf = np.arange(0, 30, tr)
hrf_signal = hrf(t_hrf)
hrf_signal /= np.max(hrf_signal)  # Normalize

# Generate neural activity for normal condition
np.random.seed(42)
neural_normal = np.random.randn(n_rois, n_timepoints)
common_signal = np.random.randn(1, n_timepoints)
neural_normal[0] += common_signal[0]  # Motor ROI
neural_normal[1] += common_signal[0]  # Sensory ROI (correlated)

# Generate neural activity for Alzheimer's condition
neural_ad = np.random.randn(n_rois, n_timepoints)
neural_ad[0] += 0.3 * common_signal[0]  # Motor ROI (reduced correlation)
neural_ad[1] += 0.3 * common_signal[0]  # Sensory ROI (reduced correlation)

# Convolve with HRF to get BOLD signals
bold_normal = np.array([convolve(n, hrf_signal, mode='same') for n in neural_normal])
bold_ad = np.array([convolve(n, hrf_signal, mode='same') for n in neural_ad])

# Add noise
noise_level = 0.5
bold_normal += noise_level * np.random.randn(*bold_normal.shape)
bold_ad += noise_level * np.random.randn(*bold_ad.shape)

# Standardize BOLD signals
bold_normal = zscore(bold_normal, axis=1)
bold_ad = zscore(bold_ad, axis=1)

# Compute correlation matrices
corr_normal = np.corrcoef(bold_normal)
corr_ad = np.corrcoef(bold_ad)

# Plot connectivity matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(corr_normal, cmap='viridis', vmin=-1, vmax=1)
axes[0].set_title('Normal Connectivity')
axes[1].imshow(corr_ad, cmap='viridis', vmin=-1, vmax=1)
axes[1].set_title('Alzheimer\'s Connectivity')
for ax in axes:
    ax.set_xticks(range(n_rois))
    ax.set_yticks(range(n_rois))
    ax.set_xticklabels(roi_names, rotation=45)
    ax.set_yticklabels(roi_names)
plt.tight_layout()
plt.savefig('connectivity_matrices.png')
plt.close()

# Plot connectivity graphs
def plot_connectivity_graph(corr_matrix, title):
    G = nx.Graph()
    for i in range(n_rois):
        G.add_node(roi_names[i])
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            if abs(corr_matrix[i, j]) > 0.5:
                G.add_edge(roi_names[i], roi_names[j], weight=corr_matrix[i, j])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(title)
    plt.savefig(f'{title.lower().replace(" ", "_")}_graph.png')
    plt.close()

plot_connectivity_graph(corr_normal, 'Normal Connectivity Graph')
plot_connectivity_graph(corr_ad, 'Alzheimer\'s Connectivity Graph')

print("Analysis complete. Check output images.")