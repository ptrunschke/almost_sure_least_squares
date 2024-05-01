import pytest
import numpy as np


def create_horribasis(dimension: int, domain: tuple[float, float], intervals: int, rng: np.random.Generator):
    assert dimension > 0
    assert len(domain) == 2 and domain[0] < domain[1]
    assert intervals > 0
    assert isinstance(rng, np.random.Generator)

    # basis_coefficients = rng.choice((-1, 1), size=(intervals, dimension))
    # s = (-dimension, -0.1, 0.1, dimension)
    # p = (1 - s[3]**2) / (s[2]**2 - s[3]**2)
    # p = ((1-p)/2, p/2, p/2, (1-p)/2)
    # basis_coefficients = rng.choice(s, size=(intervals, dimension), p=p)
    # basis_coefficients = rng.uniform(low=-1, high=1, size=(intervals, dimension))
    basis_coefficients = np.empty((intervals, dimension))
    xs = np.linspace(*domain, num=intervals)
    for j in range(dimension):
        basis_coefficients[:, j] = 1 - np.sin((j+1) * np.pi * xs)**2
    basis_coefficients, _ = np.linalg.qr(basis_coefficients, mode="reduced")
    G = basis_coefficients.T @ basis_coefficients
    assert np.allclose(G, np.eye(dimension))

    # Assuming that we integrate with respect to a uniform measure, every interval has mass 1 / intervals.
    normalisation = np.sqrt(intervals)
    def evaluate_basis(points, coefficients):
        assert points.ndim == 1 and coefficients.ndim <= 2
        assert coefficients.shape[0] <= dimension
        indices = (points - domain[0]) / (domain[1] - domain[0]) * intervals
        assert np.all(indices >= 0) and np.all(indices <= intervals)
        indices = np.minimum(indices, intervals-1).astype(int)
        return normalisation * (basis_coefficients[indices, :coefficients.shape[0]] @ coefficients).T

    return evaluate_basis


def matrix_pos(matrix):
    es, vs = np.linalg.eigh(matrix)
    es = np.maximum(es, 0)
    return vs * es @ vs.T


def draw_dpp_sample(rng, bs, sample_indices=None):
    dimension, num_nodes = bs.shape
    if sample_indices is None:
        sample_indices = []
    assert isinstance(sample_indices, list) and len(sample_indices) <= dimension
    if len(sample_indices) == dimension:
        return sample_indices
    b_factor = bs[:, sample_indices]
    b_projection = b_factor @ np.linalg.solve(b_factor.T @ b_factor, b_factor.T)
    b_projection = matrix_pos(np.eye(dimension) - b_projection)
    b_ch = np.einsum("dx, de, ex -> x", bs, b_projection, bs)
    assert np.all(b_ch >= -1e-12)
    b_ch = np.maximum(b_ch, 0)
    pdf = b_ch / np.sum(b_ch)
    sample_indices.append(rng.choice(num_nodes, p=pdf))
    return draw_dpp_sample(rng, bs, sample_indices)


def draw_repeated_dpp_sample(rng, bs, size):
    assert size > 0
    assert isinstance(rng, np.random.Generator)
    sample_indices = []
    while len(sample_indices) < size:
        sample_indices.extend(draw_dpp_sample(rng, bs))
    return sample_indices


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    from legendre import hk_gramian, orthonormal_basis

    dimension = 8
    domain = (-1, 1)
    intervals = 1000
    rng = np.random.default_rng(0)

    basis = create_horribasis(dimension, domain, intervals, rng)
    # basis = orthonormal_basis(hk_gramian(dimension, 0))

    xs = np.linspace(*domain, num=1000)
    measures = basis(xs, np.eye(dimension))
    assert measures.shape == (dimension, len(xs))

    plt.matshow(measures @ measures.T / len(xs))
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots(2, 4)
    sample_indices = []
    for k in range(dimension):
        sample_features = measures[:, sample_indices]
        projection = sample_features @ np.linalg.solve(sample_features.T @ sample_features, sample_features.T)
        projection = np.eye(dimension) - projection
        # plt.matshow(projection)
        # plt.colorbar()
        # plt.show()
        ks = np.einsum("dx, de, ex -> x", measures, projection, measures)
        assert np.all(ks >= -1e-12)
        ks = np.maximum(ks, 0)
        ks /= np.sum(ks)
        sample_indices.append(rng.choice(len(xs), p=ks))
        ax.ravel()[k].plot(xs, ks * len(xs), color="tab:blue")
        for i in sample_indices:
            ax.ravel()[k].axvline(xs[i], color="tab:red", linestyle="--")
    max_ylim = max(axi.get_ylim()[1] for axi in ax.ravel())
    for axi in ax.ravel():
        axi.set_ylim(0, max_ylim)
    plt.show()

    G = measures[:, sample_indices] @ measures[:, sample_indices].T / len(sample_indices)
    plt.plot(np.sort(np.linalg.eigvalsh(G))[::-1])
    plt.yscale("log")
    plt.show()


    # Plot phase diagram
    import matplotlib as mpl
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"""
        \usepackage{amssymb}
        \usepackage{amsmath}
        \usepackage{bbm}
    """,
    })

    # max_dimension = 100
    # max_samples = 1000
    max_dimension = 50
    max_samples = 500
    trials = 10

    step_dimension = max_dimension // 10
    step_samples = max_samples // 15
    ds = np.arange(1, max_dimension + step_dimension, step_dimension)
    ns = np.arange(1, max_samples + step_samples, step_samples)

    Ds, Ns = np.meshgrid(ds, ns)
    # Ps = rng.uniform(size=Ds.T.shape, low=0, high=1)
    Ps = np.zeros((len(ds), len(ns)))

    basis = create_horribasis(max_dimension, domain, intervals, rng)
    # basis = orthonormal_basis(hk_gramian(max_dimension, 0))

    xs = np.linspace(*domain, num=1000)
    measures = basis(xs, np.eye(max_dimension))
    assert measures.shape == (max_dimension, len(xs))
    for trial in range(trials):
        for j, d in enumerate(ds):
            ks = np.mean(measures[:d]**2, axis=0)
            sample_indices = draw_repeated_dpp_sample(rng, measures[:d], size=max_samples)
            sample_weights = 1 / ks[sample_indices]
            for i, n in enumerate(ns):
                if n < d:
                    continue
                print(f"Computing for trial={trial}, d={d}, n={n}")
                sampled_measures = measures[:d, sample_indices[:n]]
                gramian = sampled_measures * sample_weights[:n] @ sampled_measures.T / n
                lambda_min = np.min(np.linalg.eigvalsh(gramian))
                print(f"    λmin = {lambda_min:.2f}")
                Ps[j, i] += lambda_min >= 0.5
                if lambda_min >= 0.6:
                    Ps[j, i+1:] += 1
                    break

    Ps /= trials

    fig, ax = plt.subplots(1, 1)

    cm = mpl.colormaps["viridis"]
    cm = mpl.colors.LinearSegmentedColormap.from_list("grassland", cm(np.linspace(0, 0.85, 256)))
    cm.set_bad(color=cm(0))


    # The grid orientation follows the MATLAB convention: an array C with shape (nrows, ncolumns) is plotted with the column
    # number as X and the row number as Y, increasing up; hence if you have: C = rand(len(x), len(y)) then you need to transpose C.
    # vmin = 1e-2
    # vmax = 1
    # nm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    nm = mpl.colors.LogNorm()
    sc = ax.scatter(Ds, Ns, c=Ps.T, s=12, edgecolors="k", linewidths=0.5, norm=nm, cmap=cm, zorder=2)
    cbar = fig.colorbar(sc)
    # cbar.set_label(r"probability", rotation=270, labelpad=15)
    cbar.minorticks_off()

    boundary_index = np.count_nonzero(Ps <= 1e-1, axis=1) - 1
    boundary_index = np.maximum(boundary_index, 0)
    assert boundary_index.shape == ds.shape
    boundary = ns[boundary_index]  # The boundary of the phase transition.
    # Assume that boundary == slope * ds + intercept.
    slope = np.mean(np.diff(boundary) / np.diff(ds))
    print(f"Linear slope = {slope:.2f}")
    line = slope * ds
    line -= line[0]
    ax.plot(ds, line, lw=1, ls=(0, (4,2)), color="xkcd:black", zorder=1, label=r"$\sim d$")

    # # Assume that boundary == slope * ds * np.log(ds) + intercept.
    # slope = np.mean(np.diff(boundary) / np.diff(ds * np.log(ds)))
    # print(f"Log*linear slope = {slope:.2f}")
    # line = slope * ds * np.log(ds)
    # line -= line[0]
    # ax.plot(ds, line, lw=1, ls=(0, (4,2)), color="xkcd:black", zorder=1, label=r"$\sim d \log(d)$")

    ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$n$", rotation=0, labelpad=10)
    ax.set_xticks(ds)
    ax.set_title(r"Probability of $\lambda_{\mathrm{min}}(G) \ge \tfrac12$")

    plt.show()
