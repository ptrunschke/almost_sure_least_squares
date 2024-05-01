import numpy as np
from rkhs import rkhs_kernel_matrix
from legendre import hk_gramian, orthonormal_basis


def dpp_kernel_matrix(points, dimension, basis=None):
    points = np.asarray(points)
    assert points.ndim == 1
    K = rkhs_kernel_matrix(points)
    assert K.shape == (len(points), len(points))
    if basis is None:
        basis = orthonormal_basis(hk_gramian(dimension, 1))
    M = basis(points, np.eye(dimension))
    assert M.shape == (dimension, len(points))
    es, vs = np.linalg.eigh(K)
    assert np.allclose(vs * es @ vs.T, K)
    M = M @ vs
    return M / es @ M.T


def mu_W(points, dimension, *, basis=None):
    # es = np.linalg.eigvalsh(dpp_kernel_matrix(points, dimension, basis))
    # es = np.maximum(es, 0)
    # return 1 / np.sqrt(np.min(es))
    return 1 / np.sqrt(np.linalg.norm(dpp_kernel_matrix(points, dimension, basis), ord=-2))


def quasi_optimality_constant(points, dimension, *, basis=None):
    return 1 + 2 * mu_W(points, dimension, basis=basis)


def mu_p(points, dimension, *, basis=None):
    if basis is None:
        basis = orthonormal_basis(hk_gramian(dimension, 1))
    M = basis(points, np.eye(dimension)) / np.sqrt(len(points))
    # return 1 / np.linalg.norm(M, ord=-2)  # An upper bound for mu_p, when the bias is measured in the L∞ norm.
    return np.linalg.cond(M)  # The exact value of mu_p, when the bias is measures in the empirical L2 norm.


def bias_constant(points, dimension, *, basis=None):
    return 2 * mu_p(points, dimension, basis=basis)


def optimal_least_squares(function, points, dimension, basis=None):
    if basis is None:
        basis = orthonormal_basis(hk_gramian(dimension, 1))
    points = np.asarray(points)
    assert points.ndim == 1
    f = function(points)
    assert f.shape == (len(points),)
    M = basis(points, np.eye(dimension))
    assert M.shape == (dimension, len(points))
    K = rkhs_kernel_matrix(points)
    es, vs = np.linalg.eigh(K)
    assert np.allclose(vs * es @ vs.T, K)
    assert np.all(es >= -1e-12)
    es = np.maximum(es, 0)
    K_plus = vs / es @ vs.T
    V = M @ K_plus @ M.T
    v = M @ K_plus @ f
    return np.linalg.solve(V, v)


if __name__ == "__main__":
    from pathlib import Path
    from tqdm import trange
    from plotting import plt
    from matplotlib.colors import LogNorm

    dimension = 10
    target = lambda x: np.sin(2 * np.pi * x) + np.cos(2 * dimension * np.pi * x)

    rng = np.random.default_rng(7)
    points = rng.uniform(-1, 1, (2 * dimension,))

    c1 = quasi_optimality_constant(points, dimension)
    c2 = bias_constant(points, dimension)
    print(f"Quasi-optimality: {c1:.2f}")
    print(f"Bias: {c2:.2f}")

    K = rkhs_kernel_matrix(points)
    abs_K_plus = abs(np.linalg.pinv(K))
    abs_K_plus[abs_K_plus < 1e-12 * np.max(abs_K_plus)] = 0
    cmap = plt.colormaps.get_cmap("viridis")
    cmap.set_bad(color="black")
    plt.matshow(abs_K_plus, cmap=cmap, norm=LogNorm())
    plt.colorbar()
    plt.show()

    l2_legendre = orthonormal_basis(hk_gramian(dimension, 0))
    h1_legendre = orthonormal_basis(hk_gramian(dimension, 1))
    xs = np.linspace(-1, 1, 1000)
    c = optimal_least_squares(target, points, dimension, basis=h1_legendre)
    error = target(xs) - h1_legendre(xs, c)
    node_errors = target(points) - h1_legendre(points, c)
    plt.plot(xs, error, "C0-")
    plt.plot(points, node_errors, "C0o")
    c = np.linalg.lstsq(h1_legendre(points, np.eye(dimension)).T, target(points), rcond=None)[0]
    error = target(xs) - h1_legendre(xs, c)
    plt.plot(xs, error, "C1--")
    c = np.linalg.lstsq(l2_legendre(points, np.eye(dimension)).T, target(points), rcond=None)[0]
    error = target(xs) - l2_legendre(xs, c)
    plt.plot(xs, error, "C2-.")
    plt.show()

    rng = np.random.default_rng(7)
    trials = 5
    sample_sizes = range(dimension, 10 * dimension)
    lines = np.empty((trials, len(sample_sizes), 2))
    lines[:, :, 0] = sample_sizes
    mus = lines[:, :, 1]
    for trial in trange(trials):
        points = rng.uniform(-1, 1, sample_sizes[-1])
        for step, sample_size in enumerate(sample_sizes):
            mus[trial, step] = mu_W(points[:sample_size], dimension, basis=h1_legendre)

    assert np.all(mus[:, :-1] >= mus[:, 1:] - 1e-12)

    plot_directory = Path(__file__).parent / "plot"
    plot_directory.mkdir(exist_ok=True)
    plot_path = plot_directory / f"quasi-optimality_factor.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    for trial in range(trials):
        ax.plot(sample_sizes, mus[trial], label=f"Trial {trial+1}")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size")
    ax.set_title("Quasi-optimality factor")
    print("Saving sample statistics plot to", plot_path)
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )
    plt.close(fig)
