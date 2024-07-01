from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

integrals = np.load("integrals.npy")
dimension = len(integrals)

plt.style.use('seaborn-v0_8-deep')
fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
ax.violinplot(list(integrals), positions=np.arange(len(integrals)))
# ax.plot(np.arange(dimension), dimension - np.arange(dimension), "o", color="tab:red")
ax.set_xlabel("$i$")
ax.set_ylabel("$Z_i$", rotation=0)

plot_directory = Path(__file__).parent / "plot"
plot_directory.mkdir(exist_ok=True)
plot_path = plot_directory / "Z_integral_plot.png"
print("Saving Z integral statistics plot to", plot_path)
plt.savefig(
    plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
)
plt.close(fig)
