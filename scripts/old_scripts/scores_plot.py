

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime, timezone

UTCNOW = datetime.now(timezone.utc).strftime("%y%m%d.%H%M%S")

plt.close()
paths = {}
# paths[f"meanMLP (Reference)"] = f"/data/users2/ppopov1/mlp-project/assets/logs/NEW_ICA-exp-mlp_defHP-fbirn/runs.csv"
# paths[f"DBN with GTA"] = f"/data/users2/ppopov1/glass_proj/assets/logs/new_ica-exp-DBNglassReconstruct_defHP-fbirn/runs.csv"
# paths[f"meanDBN"] = f"/data/users2/ppopov1/glass_proj/assets/logs/new_ica-exp-DBNglassRecMean_defHP-fbirn/runs.csv"
# paths[f"Deep attention meanDBN"] = f"/data/users2/ppopov1/glass_proj/assets/logs/new_ica-exp-DBNglassDeep_defHP-fbirn/runs.csv"
# paths[f"Deeper attention meanDBN"] = f"/data/users2/ppopov1/glass_proj/assets/logs/new_ica-exp-DBNglassDeeper_defHP-fbirn/runs.csv"
# paths[f"DeepX4 attention meanDBN"] = f"/data/users2/ppopov1/glass_proj/assets/logs/new_ica-exp-DBNglassDeepX4_defHP-fbirn/runs.csv"

paths[f"meanMLP"] = f"/data/users2/ppopov1/mlp-project/assets/logs/NEW_ICA-exp-mlp_defHP-fbirn/runs.csv"
paths[f"glassDBN"] = f"/data/users2/ppopov1/glass_proj/assets/logs/new_ica-exp-DBNglassDeeper_defHP-fbirn/runs.csv"
paths[f"glassDBN rerun"] = f"/data/users2/ppopov1/glass_proj/assets/logs/_orig-exp-DBNglassDeeper_defHP-fbirn/runs.csv"
paths[f"glassDBN fixed"] = f"/data/users2/ppopov1/glass_proj/assets/logs/_cleaned-exp-DBNglassFIX_defHP-fbirn/runs.csv"
paths[f"glassDBN denser"] = f"/data/users2/ppopov1/glass_proj/assets/logs/_denser-exp-DBNglassFIX_defHP-fbirn/runs.csv"

# List to store the data for plotting
data_list = []

# Read each CSV file and store the relevant data
for model_name, path in paths.items():
    df = pd.read_csv(path)
    # Assuming the CSV file has a column named "test_score"
    df['model'] = model_name  # Add a column with the model name
    data_list.append(df[['test_score', 'model']])

# Concatenate all the data into a single DataFrame
data = pd.concat(data_list)

# Plot the data using seaborn
plt.figure(figsize=(26, 6))
sns.boxplot(x='model', y='test_score', data=data)
plt.ylim(0.7, 0.95)
plt.title('Test Score Distribution by Model')
plt.ylabel('AUROC')
plt.xlabel('Model')


# Show the plot
plt.savefig(f"/data/users2/ppopov1/glass_proj/scripts/pictures2/model_comparison_{UTCNOW}.png")


