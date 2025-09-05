from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from plotting import plotting

data_directory = Path(__file__).parent / "plot"
plot_directory = Path(__file__).parent / "plot"
plot_directory.mkdir(parents=True, exist_ok=True)

mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')

geometry = {
    "top": 1,
    "bottom": 0,
    "left": 0,
    "right": 1,
    "wspace": 0.25,  # the default as defined in rcParams
    "hspace": 0.25,  # the default as defined in rcParams
}
figshape = (1, 3)
textwidth = 6.50127  # width of the text in inches
figwidth = textwidth  # width of the figure in inches
phi = (1 + np.sqrt(5)) / 2
aspect_ratio = 1 / phi
figsize = plotting.compute_figsize(geometry, figshape, aspect_ratio=aspect_ratio, figwidth=figwidth)
# figsize = (1.15 * figsize[0], 1.15 * figsize[1] * 1.5)

fig, ax = plt.subplots(*figshape, figsize=figsize, sharey=True, dpi=300)
ax = ax.reshape(-1)

titles = [
    r"$H^1([-1,1], \tfrac{1}{2}\mathrm{d}x)$",
    r"$H^1_0([-1,1], \tfrac{1}{2}\mathrm{d}x)$",
    r"$H^1(\mathbb{R}, \mathcal{N}(0, 1))$",
]
for e, space in enumerate(["h1", "h10", "h1gauss"]):
    data_path = data_directory / f"integrals_{space}_embedding_sampling.npy"
    print(f"Loading sample statistics from '{data_path}'")
    integrals = np.load(data_path)
    dimension = len(integrals)
    print(f"Plotting to axes {e}")
    violin = ax[e].violinplot(list(integrals), positions=np.arange(1, len(integrals)+1))
    violin['cbars'].set_linewidth(1)
    violin['cmins'].set_linewidth(1)
    violin['cmaxes'].set_linewidth(1)
    ax[e].set_xlabel("$i$")
    ax[e].set_xticks(np.arange(1, dimension+1, 1))
    ax[e].set_yticks(np.arange(1, dimension+1, 1))
    ax[e].set_title(titles[e])
    ax[e].yaxis.grid()
ax[0].set_ylabel("$Z_i$", rotation=0)

fig.tight_layout()

plot_path = plot_directory / "Z_integral_plot.png"
print(f"Saving Z integral statistics plot to '{plot_path}'")
plt.savefig(plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True)
plt.close(fig)
