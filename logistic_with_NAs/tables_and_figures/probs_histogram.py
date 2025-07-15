#####
#
# Plot histogram of P[Y=1 | complete X]
#
#####

# %% load packages

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

exp = "SimMCAR"

setup_df = pd.read_csv(os.path.join("data", exp, "set_up.csv"))

# %% 


all_probs = []

for i in range(len(setup_df)):

    setup = setup_df.iloc[i]["set_up"]
    data = np.load(os.path.join("data", exp, "original_data", f"{setup}.npz"))

    y_probs = data["y_probs"]

    all_probs.append(y_probs)


# %%

plt.figure(figsize=(8, 4))

for i, probs in enumerate(all_probs):
    sns.kdeplot(probs, label=f'Replicate {i+1}', linewidth=2, clip=(0, 1))

plt.title(r'Kernel Density Estimates of Probabilities of Y given X', fontsize=14)
plt.xlabel('Probability', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlim(0, 1) # Explicitly set x-axis limits to 0 and 1 for clarity
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Place legend outside to avoid obscuring plot
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig("plots_scripts/plots/SimA_probs_histogram.pdf")
plt.show()


