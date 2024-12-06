

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.close()

# Define the dictionary of paths
# paths = {
#     # "New model": "/data/users2/ppopov1/glass_proj/assets/logs/test-exp-dice_3_defHP-fbirn/k_00/trial_0000/",
#     # "Orig DICE": "/data/users2/ppopov1/glass_proj/assets/logs/test-exp-dice_defHP-fbirn/k_00/trial_0000/",
#     # "Newer model": "/data/users2/ppopov1/glass_proj/assets/logs/test-exp-dice_4_defHP-fbirn/k_00/trial_0000/",
#     # "LeakyReLU model": "/data/users2/ppopov1/glass_proj/assets/logs/test-exp-dice_5_defHP-fbirn/k_00/trial_0000/",
#     "Debug model": "/data/users2/ppopov1/glass_proj/assets/logs/test-exp-dice_5_defHP-fbirn/k_00/trial_0000/",
# }

savepath = "/data/users2/ppopov1/glass_proj/scripts/debug/"

# Number of samples to plot
window = 10
end_idx = 47
start_idx = end_idx - window + 1

# Create a figure with n_samples_to_plot columns and len(paths) rows

def check_nan(arr):
    return np.any(np.isnan(arr))

def check_inf(arr):
    return np.any(np.isinf(arr))

suss = []
arr = torch.load(savepath+f"h_attn_{end_idx}.pt").cpu().detach().numpy()
for i, ar in enumerate(arr):
    if check_nan(ar):
        suss.append(i)
        break

print("failed samples in the batch: ", suss)
sus = suss[0]
sus = 45

# Plot the heatmaps
fig, axes = plt.subplots(3, window+1, figsize=(5 * (window), 5 * 3))
fig.suptitle(f"Time flow - left to right. Last column: where nans appear")
for j in range(window):
    data_idx = start_idx + j
    # arr_gru = np.transpose(torch.load(savepath+f"h_gru_{data_idx}.pt").cpu().detach().numpy(), (1, 0, 2, 3))[sus][0]
    arr_attn = torch.load(savepath+f"h_attn_{data_idx}.pt").cpu().detach().numpy()[sus][0]
    align_matrix = torch.load(savepath+f"align_matrix_{data_idx}.pt").cpu().detach().numpy()[sus]

    # vlim = max(abs(arr_gru.min()), abs(arr_gru.max()))
    # sns.heatmap(arr_gru, ax=axes[0, j], cmap='seismic', cbar=True, vmin=-vlim, vmax=vlim)

    vlim = max(abs(arr_attn.min()), abs(arr_attn.max()))
    sns.heatmap(arr_attn, ax=axes[1, j], cmap='seismic', cbar=True, vmin=-vlim, vmax=vlim)

    vlim = max(abs(align_matrix.min()), abs(align_matrix.max()))
    sns.heatmap(align_matrix, ax=axes[2, j], cmap='seismic', cbar=True, vmin=-vlim, vmax=vlim, square=True)

    axes[0, j].set_xticks([])
    axes[0, j].set_yticks([])
    axes[1, j].set_xticks([])
    axes[1, j].set_yticks([])
    axes[2, j].set_xticks([])
    axes[2, j].set_yticks([])


# Adjust the layout
plt.tight_layout(rect=[0.05, 0, 1, 0.96])

# Save the figure as a PNG file
path = "/data/users2/ppopov1/glass_proj/scripts/pictures/"
plt.savefig(path + "_debug.png")

# Show the plot
plt.close()