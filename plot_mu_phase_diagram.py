# -*- coding: utf-8 -*-
"""
Plot the recovery phase diagram: sample size against dimension.
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import colorcet  # noqa

from plotting import plotting

mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')

samplings = ["christoffel_sampling", "volume_sampling", "embedding_sampling"]
sampling_names = ["Christoffel sampling", "Volume sampling", "SIVS (our method)"]
spaces = ["h10", "h1", "h1gauss"]
ZERO = 1e-3

# cm = mpl.colormaps['viridis']
# cm = mpl.colors.LinearSegmentedColormap.from_list("grassland", cm(np.linspace(0, 0.85, 256)))
cm = plt.get_cmap("cet_gray")

vmin = ZERO
vmax = 1
nm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

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
# figwidth = textwidth / 2  # width of the figure in inches
phi = (1 + np.sqrt(5)) / 2
aspect_ratio = 0.85
figsize = plotting.compute_figsize(geometry, figshape, aspect_ratio=aspect_ratio, figwidth=figwidth)
figsize = (figsize[0], figsize[1] * 1.5)

for space in spaces:
    ds_end = 15 if space == "h1gauss" else 20
    fig, ax = plt.subplots(*figshape, figsize=figsize, sharey=True, dpi=300)
    for e, sampling in enumerate(samplings):
        data_path = Path(f"plot/suboptimality_constants_{space}_{sampling}.npz")
        print(f"Loading sample statistics from '{data_path}'")
        z = np.load(data_path)

        ds = z["ds"]
        assert ds[0] == 1 and ds[-1] == ds_end
        ns = z["ss"]
        _ds, _ns = np.meshgrid(ds, ns)
        # The grid orientation follows the MATLAB convention: an array C with shape (nrows, ncolumns) is plotted with the column
        # number as X and the row number as Y, increasing up; hence if you have: C = rand(len(x), len(y)) then you need to transpose C.
        ps = np.maximum(np.mean(z["mus"] <= 2, axis=-1), ZERO).T
        # ps = ps > 0.5  # NOTE: uncomment to find the appropriate rates
        lws = np.full(ps.shape, 0.4)
        lws[ps > 0.5] = 0.8
        ecs = np.zeros(ps.shape + (3,))
        ecs[:] = plotting.mix("k")[:3]
        # ecs[ps > 0.5, :] = plotting.mix(cm(256), 45, "k")[:3]
        ecs[ps > 0.5, :] = plotting.mix("tab:red")[:3]
        ax[e].scatter(_ds.reshape(-1), _ns.reshape(-1), c=ps.reshape(-1), s=12, edgecolors=ecs.reshape(-1, 3), linewidths=lws.reshape(-1), norm=nm, cmap=cm, zorder=2)
        ax[e].set_xlabel(r'$d$')
        ax[e].set_xticks(np.arange(5, ds[-1]+1, 5))
        ax[e].set_title(sampling_names[e])
    ax[0].set_ylabel(r'$n$', rotation=0, labelpad=10)
    if ds[-1] == 15:
        ax[0].set_yticks(np.arange(10, 60, 20))
    else:
        ax[0].set_yticks(np.arange(10, 80, 20))

    def plot_rate(axes, rate, line_style):
        ylim = axes.get_ylim()
        dss = np.linspace(ds[0], ds[-1], 100)
        rss = rate(dss)
        axes.plot(dss, rss, lw=1, ls=line_style, color="k", zorder=1, label=r"$d\log(d)$")
        axes.set_ylim(*ylim)

    if space == "h1":
        lss = [(0, (8,2)), (0, (4,2)), (0, (2,2))]
        plot_rate(ax[0], lambda x: x * np.log(x) / 1.1 + 1, lss[0])
        plot_rate(ax[1], lambda x: x**2 / 2.5 + 0.6, lss[1])
        plot_rate(ax[2], lambda x: 1.5 * x - 0.5, lss[2])
        lines = [Line2D([], [], lw=1, ls=ls, color="k") for ls in lss]
        labels = [r"$\mathcal{O}(d\log(d))$", r"$\mathcal{O}(d^2)$", r"$\mathcal{O}(d)$"]
        fig.legend(lines, labels, loc='outside upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    elif space == "h10":
        lss = [(0, (8,2)), (0, (4,2)), (0, (2,2))]
        plot_rate(ax[0], lambda x: x * np.log(x) / 1.1 + 1, lss[0])
        plot_rate(ax[1], lambda x: x**2 / 3.75 + (1 - 1/3.75), lss[1])
        plot_rate(ax[2], lambda x: 1.5 * x - 0.5, lss[2])
        lines = [Line2D([], [], lw=1, ls=ls, color="k") for ls in lss]
        labels = [r"$\mathcal{O}(d\log(d))$", r"$\mathcal{O}(d^2)$", r"$\mathcal{O}(d)$"]
        fig.legend(lines, labels, loc='outside upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    else:
        assert space == "h1gauss"
        lss = [(0, (8,2)), (0, (4,2)), (0, (2,2))]
        plot_rate(ax[0], lambda x: x * np.log(x) / 0.95 + 1, lss[0])
        plot_rate(ax[1], lambda x: 2.1 * x - (2.1 - 1), lss[1])
        plot_rate(ax[2], lambda x: 1.2 * x - (1.2 - 1), lss[2])
        lines = [Line2D([], [], lw=1, ls=ls, color="k") for ls in lss]
        labels = [r"$\mathcal{O}(d\log(d))$", r"$\mathcal{O}(d)$", r"$\mathcal{O}(d)$"]
        fig.legend(lines, labels, loc='outside upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    fig.tight_layout()

    # plot_path = Path(f"plot/suboptimality_constants_{space}.png")
    plot_path = Path(f"new_plots/suboptimality_constants_{space}.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving sample statistics plot to '{plot_path}'")
    plt.savefig(plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True)
    plt.close(fig)
