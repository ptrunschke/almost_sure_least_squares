from pathlib import Path
import pytest
import numpy as np
from rkhs import rkhs_kernel_matrix
from legendre import hk_gramian, orthonormal_basis


def mercer_decomposition(points, rank):
    K = rkhs_kernel_matrix(points)
    es, vs = np.linalg.eigh(K)
    es, vs = es[::-1], vs[:, ::-1]
    assert np.allclose(vs * es @ vs.T, K)
    assert np.all(es[:-1] >= es[1:])
    fs = (vs[:, :rank] * np.sqrt(es[:rank])).T
    assert fs.shape == (rank, len(points))
    return fs


def test_mercer_decomposition():
    xs = np.linspace(-1, 1, 1_000)
    R = len(xs)
    fs = mercer_decomposition(xs, R)
    assert fs.shape == (R, len(xs))
    assert np.allclose(fs.T @ fs, rkhs_kernel_matrix(xs))


if __name__ == "__main__":
    from plotting import plt

    dimension = 10
    # NOTE: The quality of the samples is not very good for small R.
    # R = 1 * dimension
    # R = 10 * dimension
    R = 100 * dimension

    alpha = 1e-2
    assert alpha == float(f"{alpha:.0e}")

    plot_directory = Path(__file__).parent / "plot" / f"R-{R}_α-{alpha:.0e}"
    plot_directory.mkdir(exist_ok=True)

    xs = np.linspace(-1, 1, 1000)
    fs = mercer_decomposition(xs, R)

    plot_path = plot_directory / f"eigenvalues.png"
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    sigmas = np.trapz(fs**2, xs, axis=1) / 2
    ax[0].plot(sigmas)
    ax[0].set_yscale("log")
    ax[0].set_title("Eigenvalues of the kernel")
    for k in range(5):
        ax[1].plot(xs, fs[k], label=f"$f_{{{k}}}$")
    ax[1].legend()
    ax[1].set_title("Eigenfunctions of the kernel")
    print("Saving eigenvalue plot to", plot_path)
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )
    plt.close(fig)

    h1_legendre = orthonormal_basis(hk_gramian(dimension, 1))
    bs = h1_legendre(xs, np.eye(dimension))
    assert bs.shape == (dimension, len(xs))

    plot_path = plot_directory / f"christoffel.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    ch_b = np.sum(bs**2, axis=0)
    ch_b /= np.trapz(ch_b, xs)
    ch_f = np.sum(fs**2, axis=0)
    ch_f /= np.trapz(ch_f, xs)
    ch_bf = ch_b / ch_f
    ch_bf /= np.trapz(ch_bf, xs)
    ax.plot(xs, ch_b, label=r"$\mathfrak{K}_{\mathcal{V}_d}$")
    ax.plot(xs, ch_f, label=r"$\mathfrak{K}_{\mathcal{V}}$")
    ax.plot(xs, ch_bf, label=r"$\frac{\mathfrak{K}_{\mathcal{V}_d}}{\mathfrak{K}_{\mathcal{V}}}$")
    ax.legend()
    ax.set_title("Christoffel densities")
    print("Saving Christoffel plot to", plot_path)
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )
    plt.close(fig)


if __name__ == "__main__" and __debug__:
    d = 3
    X = np.random.randn(d+2, d)
    det = np.linalg.det(X.T @ X)
    prec_det = 1
    while X.shape[1] > 0:
        x1 = X[:, 0]
        X = X[:, 1:]
        projection = X @ np.linalg.pinv(X.T @ X) @ X.T
        projection_det = x1 @ x1 - x1 @ projection @ x1
        prec_det *= projection_det
        remaining_det = np.linalg.det(X.T @ X)
        assert np.isclose(prec_det * remaining_det, det)
    assert np.allclose(projection, 0)


def matrix_pos(matrix):
    es, vs = np.linalg.eigh(matrix)
    es = np.maximum(es, 0)
    return vs * es @ vs.T


def draw_sample(rng, bs, fs, plot=False, sample_indices=None):
    dimension, num_nodes = bs.shape
    rank = fs.shape[0]
    assert fs.shape == (rank, num_nodes)
    if sample_indices is None:
        sample_indices = []
    assert isinstance(sample_indices, list) and len(sample_indices) <= dimension <= rank
    if len(sample_indices) == dimension and not plot:
        return sample_indices
    f_factor = fs[:, sample_indices]
    f_projection = f_factor @ np.linalg.solve(f_factor.T @ f_factor, f_factor.T)
    f_projection = matrix_pos(np.eye(rank) - f_projection)
    f_ch = np.einsum("dx, de, ex -> x", fs, f_projection, fs)
    assert np.all(f_ch >= -1e-12)
    f_ch = np.maximum(f_ch, 0)
    b_factor = bs[:, sample_indices]
    b_projection = b_factor @ np.linalg.solve(b_factor.T @ b_factor, b_factor.T)
    b_projection = matrix_pos(np.eye(dimension) - b_projection)
    b_ch = np.einsum("dx, de, ex -> x", bs, b_projection, bs)
    assert np.all(b_ch >= -1e-12)
    b_ch = np.maximum(b_ch, 0)
    f_ch /= np.sum(f_ch)
    f_ch = np.maximum(f_ch, alpha / num_nodes)
    pdf = b_ch / f_ch
    if plot:
        plot_path = plot_directory / f"sampling_density_step-{len(sample_indices)+1}.png"
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
        ax.plot(pdf / np.sum(pdf), color="tab:blue")
        ax.plot(b_ch / np.sum(b_ch), color="tab:orange")
        for idx in sample_indices:
            ax.axvline(idx, color="tab:red")
        if len(sample_indices) == dimension:
            ax.set_title(f"Optimal sampling distribution")
        else:
            ax.set_title(f"Optimal sampling distribution (step {len(sample_indices)+1})")
        print("Saving optimal sampling plot to", plot_path)
        plt.savefig(
            plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
        )
        plt.close(fig)
    if len(sample_indices) == dimension:
        return sample_indices
    pdf /= np.sum(pdf)
    sample_indices.append(rng.choice(num_nodes, p=pdf))
    return draw_sample(rng, bs, fs, plot, sample_indices)


class Sampler(object):
    def __init__(self, dimension, rank, discretisation) -> None:
        self.xs = np.linspace(-1, 1, discretisation)
        self.fs = mercer_decomposition(self.xs, rank)
        self.basis = orthonormal_basis(hk_gramian(dimension, 1))
        self.bs = self.basis(self.xs, np.eye(dimension))

    @property
    def discretisation(self):
        return len(self.xs)

    @property
    def rank(self):
        return self.fs.shape[0]

    @property
    def dimension(self):
        return self.bs.shape[0]

    def draw(self, rng):
        indices = draw_sample(rng, self.bs, self.fs)
        return self.xs[indices]


if __name__ == "__main__":
    from least_squares import quasi_optimality_constant, bias_constant, optimal_least_squares

    target = lambda x: np.sin(2 * np.pi * x) + np.cos(2 * dimension * np.pi * x)

    rng = np.random.default_rng(0)
    idcs = draw_sample(rng, bs, fs, plot=True)
    points = xs[idcs]

    c1 = quasi_optimality_constant(points, dimension)
    c2 = bias_constant(points, dimension)
    print(f"Quasi-optimality: {c1:.2f}")
    print(f"Bias: {c2:.2f}")

    plot_path = plot_directory / f"error.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    l2_legendre = orthonormal_basis(hk_gramian(dimension, 0))
    h1_legendre = orthonormal_basis(hk_gramian(dimension, 1))
    xs = np.linspace(-1, 1, 1000)
    c = optimal_least_squares(target, points, dimension, basis=h1_legendre)
    error = target(xs) - h1_legendre(xs, c)
    node_errors = target(points) - h1_legendre(points, c)
    ax.plot(xs, error, "C0-", label="Optimal approximation")
    ax.plot(points, node_errors, "C0o")
    c = np.linalg.lstsq(h1_legendre(points, np.eye(dimension)).T, target(points), rcond=None)[0]
    error = target(xs) - h1_legendre(xs, c)
    ax.plot(xs, error, "C1--", label="$H^1$ least squares")
    c = np.linalg.lstsq(l2_legendre(points, np.eye(dimension)).T, target(points), rcond=None)[0]
    error = target(xs) - l2_legendre(xs, c)
    ax.plot(xs, error, "C2-.", label="$L^2$ least squares")
    ax.legend()
    ax.set_title("Pointwise error")
    print("Saving error plot to", plot_path)
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )
    plt.close(fig)

    from tqdm import trange
    trials = 1_000
    samples = []
    for trial in trange(trials):
        samples.append(draw_sample(rng, bs, fs))

    plot_path = plot_directory / f"sample_statistics.png"
    fig, ax = plt.subplots(1, 3, figsize=(8, 4), dpi=300)
    c1s = []
    c2s = []
    for e, sample in enumerate(samples, start=1):
        c1s.append(quasi_optimality_constant(xs[sample], dimension, h1_legendre))
        c2s.append(bias_constant(xs[sample], dimension, h1_legendre))
    ax[1].hist(c1s, bins=25)
    ax[1].set_title("Quasi-optimality factor")
    ax[2].hist(c2s, bins=25)
    ax[2].set_title("Noise amplification factor")
    samples = np.concatenate(samples)
    ax[0].hist(xs[samples], density=True, bins=80)
    ax[0].set_title("Sample distribution")
    print("Saving sample statistics plot to", plot_path)
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )
    plt.close(fig)
