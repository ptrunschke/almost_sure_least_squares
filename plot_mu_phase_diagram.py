# -*- coding: utf-8 -*-
"""
"""

"""
Plot the recovery phase diagram: order-vs-radius.

The result is written to ./figures.
"""
from pathlib import Path
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(dest='DATA', help="file containing the data")
args = parser.parse_args()

mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')

z = np.load(args.DATA)
ds = z["ds"]
ss = z["ss"]
_ds, _ss = np.meshgrid(ds, ss)
mus = z["mus"]
mean_mus = np.mean(mus <= 2, axis=-1)
zero = 1e-3
mean_mus[mean_mus == 0] = zero

cm = mpl.colormaps['viridis']
cm = mpl.colors.LinearSegmentedColormap.from_list("grassland", cm(np.linspace(0, 0.85, 256)))

# The grid orientation follows the MATLAB convention: an array C with shape (nrows, ncolumns) is plotted with the column
# number as X and the row number as Y, increasing up; hence if you have: C = rand(len(x), len(y)) then you need to transpose C.
vmin = zero
vmax = 1
nm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)
sc = ax.scatter(_ds, _ss, c=mean_mus.T, s=12, edgecolors='k', linewidths=0.5, norm=nm, cmap=cm, zorder=2)
ylim = ax.get_ylim()
dss = np.linspace(ds[0], ds[-1], 100)
# p = 0.5  # Probability of success
# nss = 2*dss*np.log(2*dss/(1-p))
nss = dss*np.log(dss)
ax.plot(dss, nss, lw=1, ls=(0, (4,2)), color='xkcd:black', zorder=1, label=r"$d\log(d)$")
ax.plot(dss, dss, lw=1, ls=(0, (2,2)), color='xkcd:black', zorder=1, label=r"$d$")
ax.set_ylim(*ylim)
ax.set_xlabel(r'$d$')
ax.set_ylabel(r'$n$', rotation=0, labelpad=10)
ax.set_title(r"$\mathbb{P}[\mu(\boldsymbol{x}) \le 2]$")

ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

plt.tight_layout()
plt.show()

# plt.subplots_adjust(**geometry)
# PLOTFILE = args.DATA.split('/')[-1][:-4]+".pdf"
# PLOTFILE = f"figures/{PLOTFILE}"
# os.makedirs("figures", exist_ok=True)
# plt.savefig(PLOTFILE, format="pdf", facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches="tight")


# plot_path = plot_directory / f"sample_statistics_{space}_{basis_name}_{oversampling}x.png"
# print("Saving sample statistics plot to", plot_path)
# plt.savefig(
#     plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
# )
# plt.close(fig)
