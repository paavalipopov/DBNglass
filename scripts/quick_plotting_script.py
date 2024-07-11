

import torch
import matplotlib.pyplot as plt
import seaborn as sns

plt.close()

# Define the dictionary of paths
paths = {
    "New model": "/data/users2/ppopov1/glass_proj/assets/logs/test-exp-dice_3_defHP-fbirn/k_00/trial_0000/DNC.pt",
    "DICE with DAG loss new": "/data/users2/ppopov1/glass_proj/assets/logs/test-exp-dice_dag_1_defHP-fbirn/k_00/trial_0000/DNCs.pt",
    "Orig DICE": "/data/users2/ppopov1/glass_proj/assets/logs/test-exp-dice_defHP-fbirn/k_00/trial_0000/DNCs.pt",
}

# Number of samples to plot
n_samples_to_plot = 3

# Create a figure with n_samples_to_plot columns and len(paths) rows
fig, axes = plt.subplots(len(paths), n_samples_to_plot, figsize=(5 * n_samples_to_plot, 5 * len(paths)))

# Plot the heatmaps
for row_idx, (title, path) in enumerate(paths.items()):
    array = torch.load(path).cpu().detach().numpy()
    
    for col_idx in range(n_samples_to_plot):
        vlim = max(abs(array[col_idx].min()), abs(array[col_idx].max()))
        sns.heatmap(array[col_idx], ax=axes[row_idx, col_idx], cmap='seismic', cbar=True, vmin=-vlim, vmax=vlim, square=True)
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
    
    # Add row title
    axes[row_idx, 0].annotate(title, xy=(0, 0.5), xytext=(-axes[row_idx, 0].yaxis.labelpad - 5, 0),
                              xycoords=axes[row_idx, 0].yaxis.label, textcoords='offset points',
                              size='large', ha='right', va='center', rotation=90)

# Adjust the layout
plt.tight_layout(rect=[0.05, 0, 1, 0.96])

# Save the figure as a PNG file
plt.savefig("+".join(paths.keys()) + ".png")

# Show the plot
plt.show()