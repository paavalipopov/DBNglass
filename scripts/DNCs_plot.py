

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.close()

paths = {}
# paths[f"Giorgio model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/GOG-exp-DBNglassGOG_defHP-fbirn/k_04/trial_0009/"
# paths[f"Deep model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Deep3-exp-DBNglassDeep_defHP-fbirn-single_HPs/k_00/trial_0001/"
# paths[f"DeepSV model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/DeepSV3-exp-DBNglassDeepSV_defHP-fbirn-single_HPs/k_04/trial_0009/"
# paths[f"Deep32LR model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Deep32LR-exp-DBNglassDeep_defHP-fbirn-single_HPs/k_00/trial_0001/"
# paths[f"DeepSV32 model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/DeepSV32-exp-DBNglassDeepSV_defHP-fbirn-single_HPs/k_04/trial_0009/"

# paths[f"Deep64 model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Deep3-exp-DBNglassDeep_defHP-fbirn-single_HPs/k_04/trial_0009/"
# paths[f"Deep32 model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Deep32-exp-DBNglassDeep_defHP-fbirn-single_HPs/k_04/trial_0009/"
# paths[f"Deep16LR model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Deep32LR-exp-DBNglassDeep_defHP-fbirn-single_HPs/k_04/trial_0009/"
# paths[f"Norm model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/MIXmean16-exp-DBNglassMIXnorm_defHP-fbirn-single_HPs/k_04/trial_0009/"
# paths[f"Gamma model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Gamma16-exp-DBNglassGamma_defHP-fbirn-single_HPs/k_00/trial_0001/"
# paths[f"Mix-mean model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Mean16LR-exp-DBNglassMean_defHP-fbirn-single_HPs/k_04/trial_0009/"
# paths[f"Reconstructing model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Rec16LR-exp-DBNglassReconstruct_defHP-fbirn-single_HPs/k_04/trial_0009/"
# paths[f"Pretrained1 model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Pretrained1-exp-DBNglassReconstruct_defHP-fbirn/k_04/trial_0009/"
# paths[f"Pretrained mean model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/RecMean-exp-DBNglassRecMean_defHP-fbirn/k_04/trial_0009/"
paths[f"Pretrained2 mean model"] = f"/data/users2/ppopov1/glass_proj/assets/logs/RecMean2-exp-DBNglassRecMean_defHP-fbirn/k_04/trial_0009/"


# for i in range(6):
#     paths[f"New Hoyer {i}"] = f"/data/users2/ppopov1/glass_proj/assets/logs/NHoyer{i}-exp-DBNglassHoyer_defHP-fbirn-single_HPs/k_04/trial_0009/"
# paths[f"Best 1"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Deep3-exp-DBNglassDeep_defHP-fbirn-single_HPs/k_02/trial_0001/"
# for i in range(4):
#     paths[f"Sparse {i}"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Gate{i}-exp-DBNglassGate_defHP-fbirn-single_HPs/k_04/trial_0009/"
# for i in range(4, 8):
#     paths[f"Sparsity and DAG {i}"] = f"/data/users2/ppopov1/glass_proj/assets/logs/NEWWW{i}-exp-DBNglassNoTan_defHP-fbirn-single_HPs/k_04/trial_0009/"
# for i in range(8):
#     paths[f"Config {i}"] = f"/data/users2/ppopov1/glass_proj/assets/logs/NEWWW{i}-exp-DBNglassNoTan_defHP-fbirn-single_HPs/k_04/trial_0009/"

# for i in range(4):
#     paths[f"No sparsity {i}"] = f"/data/users2/ppopov1/glass_proj/assets/logs/NEWWW{i}-exp-DBNglassNoTan_defHP-fbirn-single_HPs/k_04/trial_0009/"
#     paths[f"Sparsity {i}"] = f"/data/users2/ppopov1/glass_proj/assets/logs/NNEW{i}-exp-DBNglassNoDag_defHP-fbirn-single_HPs/k_04/trial_0009/"

# paths[f"No Gate"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Gate0-exp-DBNglassGate_defHP-fbirn-single_HPs/k_04/trial_0009/"
# paths[f"Gate"] = f"/data/users2/ppopov1/glass_proj/assets/logs/Gate1-exp-DBNglassGate_defHP-fbirn-single_HPs/k_04/trial_0009/"

# Number of samples to plot
n_samples_to_plot = 3
n_timepoints_to_plot = 10

# Create a figure with n_samples_to_plot columns and len(paths) rows

# Plot the heatmaps
for name, path in paths.items():
    print(name)
    try:
        fig, axes = plt.subplots(n_samples_to_plot, 1+n_timepoints_to_plot, figsize=(5 * (1 + n_timepoints_to_plot), 5 * n_samples_to_plot))
        fig.suptitle(f"{name} extended (first column - combined, others - time-specific)")
        for row_idx in range(n_samples_to_plot):
            # print("row ", row_idx)
            array = torch.load(path+"DNCs.pt").cpu().detach().numpy()[row_idx]
            maxval = np.max(np.abs(array))
            
            for col_idx in range(n_timepoints_to_plot+1):
                # print("row ", col_idx)
                if col_idx == 0:
                    continue
                    array = torch.load(path+"DNC.pt").cpu().detach().numpy()[row_idx]
                    vlim = np.max(np.abs(array))
                else:
                    if col_idx <= n_timepoints_to_plot // 2:
                        data_idx = col_idx-1
                    else:
                        data_idx =  - n_timepoints_to_plot + col_idx - 1
                    array = torch.load(path+"DNCs.pt").cpu().detach().numpy()[row_idx][data_idx].squeeze()
                    # vlim = max(abs(array.min()), abs(array.max()))
                    vlim = maxval
                
                sns.heatmap(array, ax=axes[row_idx, col_idx], cmap='seismic', cbar=True, vmin=-vlim, vmax=vlim, square=True)
                axes[row_idx, col_idx].set_xticks([])
                axes[row_idx, col_idx].set_yticks([])
                
            # Add row title
            axes[row_idx, 5].set_title(f"Time point 5")
            
            axes[row_idx, 6].set_title(f"Time point -5")

        # Adjust the layout
        plt.tight_layout(rect=[0.05, 0, 1, 0.96])

        # Save the figure as a PNG file
        plt.savefig(f"/data/users2/ppopov1/glass_proj/scripts/pictures/{name} extended.png")

        # Show the plot
        plt.close()
    except Exception as e:
        print(e)
        continue