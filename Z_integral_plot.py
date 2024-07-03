from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

integrals = np.load("plot/integrals_h10_embedding_sampling.npy")
dimension = len(integrals)

plt.style.use('seaborn-v0_8-deep')
fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
ax.violinplot(list(integrals), positions=np.arange(len(integrals)))
ax.set_xlabel("$i$")
ax.set_ylabel("$Z_i$", rotation=0)
ax.set_yticks(np.arange(1, dimension+1))
ax.yaxis.grid()

plot_directory = Path(__file__).parent / "plot"
plot_directory.mkdir(exist_ok=True)
plot_path = plot_directory / "Z_integral_plot.png"
print("Saving Z integral statistics plot to", plot_path)
plt.savefig(
    plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
)
plt.close(fig)
