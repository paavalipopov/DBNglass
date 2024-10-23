

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.close()

# Define the dictionary of paths
paths = {}
# paths[f"DBN with GTA"] = f"/data/users2/ppopov1/glass_proj/assets/logs/new_ica_test-exp-DBNglassReconstruct_defHP-fbirn/k_04/trial_0009/"
# paths[f"meanDBN"] = f"/data/users2/ppopov1/glass_proj/assets/logs/new_ica_test-exp-DBNglassRecMean_defHP-fbirn/k_04/trial_0009/"
# paths[f"Deep attention meanDBN"] = f"/data/users2/ppopov1/glass_proj/assets/logs/new_ica_test-exp-DBNglassDeep_defHP-fbirn/k_04/trial_0009/"
# paths[f"Deeper attention meanDBN"] = f"/data/users2/ppopov1/glass_proj/assets/logs/new_ica_test-exp-DBNglassDeeper_defHP-fbirn/k_04/trial_0009/"
# paths[f"DeepX4 attention meanDBN"] = f"/data/users2/ppopov1/glass_proj/assets/logs/new_ica_test-exp-DBNglassDeepX4_defHP-fbirn/k_04/trial_0009/"
paths[f"retry glassDBN"] = f"/data/users2/ppopov1/glass_proj/assets/logs/_orig-exp-DBNglassDeeper_defHP-fbirn/k_00/trial_0009/"
# paths[f"fixed glassDBN"] = f"/data/users2/ppopov1/glass_proj/assets/logs/_cleaned-exp-DBNglassFIX_defHP-fbirn/k_00/trial_0009/"

# Number of samples to plot
n_samples_to_plot = 3
n_timepoints_to_plot = 10

# Create a figure with n_samples_to_plot columns and len(paths) rows

# Plot the heatmaps
for name, path in paths.items():

    fig, axes = plt.subplots(n_samples_to_plot, 1+n_timepoints_to_plot, figsize=(5 * (1 + n_timepoints_to_plot), 5 * n_samples_to_plot))
    fig.suptitle(f"{name} extended (first column - combined, others - time-specific)")
    for row_idx in range(n_samples_to_plot):
        array = torch.load(path+"DNCs.pt", map_location=torch.device('cpu')).detach().cpu().numpy()[row_idx]
        maxval = np.abs(array).max()
        
        for col_idx in range(n_timepoints_to_plot+1):
            if col_idx == 0:
                array = torch.load(path+"DNC.pt").detach().cpu().numpy()[row_idx]
                vlim = max(abs(array.min()), abs(array.max()))
            else:
                array = torch.load(path+"DNCs.pt").detach().cpu().numpy()[row_idx][col_idx-1]
                # vlim = max(abs(array.min()), abs(array.max()))
                vlim = maxval
            
            sns.heatmap(array, ax=axes[row_idx, col_idx], cmap='seismic', cbar=True, vmin=-vlim, vmax=vlim, square=True)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
        
        # Add row title
        axes[row_idx, 0].annotate(f"Subject {row_idx}", xy=(0, 0.5), xytext=(-axes[row_idx, 0].yaxis.labelpad - 5, 0),
                                xycoords=axes[row_idx, 0].yaxis.label, textcoords='offset points',
                                size='large', ha='right', va='center', rotation=90)

    # Adjust the layout
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])

    # Save the figure as a PNG file
    plt.savefig(f"/data/users2/ppopov1/glass_proj/scripts/pictures/{name} extended.png")

    # Show the plot
    plt.close()