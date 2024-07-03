from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from plotting import plotting

mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')

integrals = np.load("plot/integrals_h10_embedding_sampling.npy")
dimension = len(integrals)

geometry = {
    "top": 1,
    "bottom": 0,
    "left": 0,
    "right": 1,
    "wspace": 0.25,  # the default as defined in rcParams
    "hspace": 0.25,  # the default as defined in rcParams
}
figshape = (1, 1)
textwidth = 6.50127  # width of the text in inches
figwidth = textwidth  # width of the figure in inches
# figwidth = textwidth / 2  # width of the figure in inches
phi = (1 + np.sqrt(5)) / 2
aspect_ratio = phi
figsize = plotting.compute_figsize(geometry, figshape, aspect_ratio=aspect_ratio, figwidth=figwidth)
# figsize = (figsize[0], figsize[1] * 1.5)

fig, ax = plt.subplots(*figshape, figsize=figsize, dpi=300)
ax.violinplot(list(integrals), positions=np.arange(len(integrals)))
ax.set_xlabel("$i$")
ax.set_ylabel("$Z_i$", rotation=0)
ax.set_yticks(np.arange(1, dimension+1))
ax.yaxis.grid()

fig.tight_layout()

plot_directory = Path(__file__).parent / "plot"
plot_directory.mkdir(exist_ok=True)
plot_path = plot_directory / "Z_integral_plot.png"
print(f"Saving Z integral statistics plot to '{plot_path}'")
plt.savefig(plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True)
plt.close(fig)
