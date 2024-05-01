import matplotlib.pyplot as plt

textcolor = "#D4D6C0"
legendcolor = "#3B3B3B"
plt.rcParams.update({
    "lines.color": textcolor,
    "patch.edgecolor": textcolor,
    "axes.edgecolor": textcolor,
    "axes.labelcolor": textcolor,
    "axes.titlecolor": textcolor,
    "xtick.color": textcolor,
    "ytick.color": textcolor,
    "text.usetex": True,
    "text.latex.preamble": r"""
    \usepackage{amssymb}
    \usepackage{amsmath}
    \usepackage{bbm}
""",
    "legend.facecolor": legendcolor,
    "legend.edgecolor": textcolor,
    "legend.labelcolor": textcolor,
})