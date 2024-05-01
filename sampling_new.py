from typing import Optional
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rkhs import rkhs_kernel
from legendre import hk_gramian, orthonormal_basis


KernelFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
BasisFunction = Callable[[np.ndarray], np.ndarray]


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


def create_subspace_kernel(basis, dimension):
    def subspace_kernel(x, y):
        x_measures = basis(x, np.eye(dimension))
        assert x_measures.shape == (dimension,) + x.shape
        y_measures = basis(y, np.eye(dimension))
        assert y_measures.shape == (dimension,) + y.shape
        return (x_measures * y_measures).sum(axis=0)

    return subspace_kernel


def draw_sample(
        rng: np.random.Generator,
        rkhs_kernel: KernelFunction,
        subspace_basis: BasisFunction,
        dimension: int,
        discretisation: np.ndarray,
        *,
        plot: bool = False,
        conditioned_on: Optional[list[float]] = None,
        regularisation: float = 1e-8,
    ) -> list[int]:
    if conditioned_on is None:
        conditioned_on = []
    discretisation, conditioned_on = np.asarray(discretisation), np.asarray(conditioned_on)
    assert discretisation.ndim == 1 and conditioned_on.ndim == 1

    def bayes_kernel_variance(kernel: KernelFunction, points: np.ndarray, conditioned_on: np.ndarray) -> np.ndarray:
        assert points.ndim == 1 and conditioned_on.ndim == 1
        kxx = kernel(points, points)
        assert kxx.shape == points.shape
        kxX = kernel(points[:, None], conditioned_on[None, :])
        assert kxX.shape == points.shape + conditioned_on.shape
        kXX = kernel(conditioned_on[:, None], conditioned_on[None, :])
        assert kXX.shape == conditioned_on.shape + conditioned_on.shape
        kXX_inv_kXx = np.linalg.lstsq(kXX, kxX.T, rcond=None)[0]
        assert kXX_inv_kXx.shape == conditioned_on.shape + points.shape
        res = np.maximum(kxx - (kxX * kXX_inv_kXx.T).sum(axis=1), 0)
        assert res.shape == points.shape
        return res

    if len(conditioned_on) < dimension:
        subspace_kernel = create_subspace_kernel(subspace_basis, dimension)

        def density(points: np.ndarray) -> np.ndarray:
            numerator = bayes_kernel_variance(subspace_kernel, points, conditioned_on)
            denominator = bayes_kernel_variance(rkhs_kernel, points, conditioned_on) + regularisation
            return numerator / denominator
    else:
        def density(points: np.ndarray) -> np.ndarray:
            bc = subspace_basis(conditioned_on, np.eye(dimension))  # TODO: It makes no sense that subspace_basis requires the eye(dimension)!
            assert bc.shape == (dimension,) + conditioned_on.shape
            kcc = rkhs_kernel(conditioned_on[:, None], conditioned_on[None, :])
            assert kcc.shape == conditioned_on.shape + conditioned_on.shape
            kcc_inv_bc = np.linalg.lstsq(kcc, bc.T, rcond=None)[0]
            assert kcc_inv_bc.shape == conditioned_on.shape + (dimension,)
            G = bc @ kcc_inv_bc
            assert G.shape == (dimension, dimension)
            kcx = rkhs_kernel(conditioned_on[:, None], points[None, :])
            assert kcx.shape == conditioned_on.shape + points.shape
            bckx = kcc_inv_bc.T @ kcx
            assert bckx.shape == (dimension,) + points.shape
            bx = subspace_basis(points, np.eye(dimension))
            assert bx.shape == (dimension,) + points.shape
            diff = bx - bckx
            assert diff.shape == (dimension,) + points.shape
            numerator = (diff * np.linalg.lstsq(G, diff, rcond=None)[0]).sum(axis=0)
            assert numerator.shape == points.shape
            numerator = np.maximum(numerator, 0)
            denominator = bayes_kernel_variance(rkhs_kernel, points, conditioned_on) + regularisation
            return 1 + numerator / denominator

    pdf = density(discretisation)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0)
    pdf /= np.sum(pdf)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
        # fig, ax = plt.subplots(1, 1)
        ax.plot(discretisation, pdf / np.trapz(pdf, discretisation))
        for index, point in enumerate(conditioned_on):
            alpha = max(0.5**(len(conditioned_on)-index-1), 0.1)
            ax.axvline(point, color="tab:red", alpha=alpha)
        ax.set_title(f"Optimal sampling distribution at step {len(conditioned_on)+1}")
        plot_path = plot_directory / f"density_step-{len(conditioned_on)+1}.png"
        print("Saving optimal sampling plot to", plot_path)
        plt.savefig(
            plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
        )
        plt.close(fig)

    return rng.choice(discretisation, p=pdf)
    # return discretisation[np.argmax(pdf)]

def draw_subspace_dpp_sample(
        rng: np.random.Generator,
        subspace_basis: BasisFunction,
        dimension: int,
        discretisation: np.ndarray,
        *,
        plot: bool = False,
        conditioned_on: Optional[list[float]] = None,
    ) -> list[int]:
    if conditioned_on is None:
        conditioned_on = []
    discretisation, conditioned_on = np.asarray(discretisation), np.asarray(conditioned_on)
    assert discretisation.ndim == 1 and conditioned_on.ndim == 1

    def bayes_kernel_variance(kernel: KernelFunction, points: np.ndarray, conditioned_on: np.ndarray) -> np.ndarray:
        assert points.ndim == 1 and conditioned_on.ndim == 1
        kxx = kernel(points, points)
        assert kxx.shape == points.shape
        kxX = kernel(points[:, None], conditioned_on[None, :])
        assert kxX.shape == points.shape + conditioned_on.shape
        kXX = kernel(conditioned_on[:, None], conditioned_on[None, :])
        assert kXX.shape == conditioned_on.shape + conditioned_on.shape
        kXX_inv_kXx = np.linalg.lstsq(kXX, kxX.T, rcond=None)[0]
        assert kXX_inv_kXx.shape == conditioned_on.shape + points.shape
        res = np.maximum(kxx - (kxX * kXX_inv_kXx.T).sum(axis=1), 0)
        assert res.shape == points.shape
        return res

    if len(conditioned_on) < dimension:
        subspace_kernel = create_subspace_kernel(subspace_basis, dimension)

        def density(points: np.ndarray) -> np.ndarray:
            return bayes_kernel_variance(subspace_kernel, points, conditioned_on)
    else:
        raise NotImplementedError()

    pdf = density(discretisation)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0)
    pdf /= np.sum(pdf)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
        # fig, ax = plt.subplots(1, 1)
        ax.plot(discretisation, pdf / np.trapz(pdf, discretisation))
        for index, point in enumerate(conditioned_on):
            alpha = max(0.5**(len(conditioned_on)-index-1), 0.1)
            ax.axvline(point, color="tab:red", alpha=alpha)
        ax.set_title(f"Optimal sampling distribution at step {len(conditioned_on)+1}")
        plot_path = plot_directory / f"subspace_dpp_density_step-{len(conditioned_on)+1}.png"
        print("Saving optimal sampling plot to", plot_path)
        plt.savefig(
            plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
        )
        plt.close(fig)

    return rng.choice(discretisation, p=pdf)
    # return discretisation[np.argmax(pdf)]


def draw_dpp_sample(
        rng: np.random.Generator,
        rkhs_kernel: KernelFunction,
        dimension: int,
        discretisation: np.ndarray,
        *,
        plot: bool = False,
        conditioned_on: Optional[list[float]] = None,
    ) -> list[int]:
    if conditioned_on is None:
        conditioned_on = []
    discretisation, conditioned_on = np.asarray(discretisation), np.asarray(conditioned_on)
    assert discretisation.ndim == 1 and conditioned_on.ndim == 1

    def bayes_kernel_variance(kernel: KernelFunction, points: np.ndarray, conditioned_on: np.ndarray) -> np.ndarray:
        assert points.ndim == 1 and conditioned_on.ndim == 1
        kxx = kernel(points, points)
        assert kxx.shape == points.shape
        kxX = kernel(points[:, None], conditioned_on[None, :])
        assert kxX.shape == points.shape + conditioned_on.shape
        kXX = kernel(conditioned_on[:, None], conditioned_on[None, :])
        assert kXX.shape == conditioned_on.shape + conditioned_on.shape
        kXX_inv_kXx = np.linalg.lstsq(kXX, kxX.T, rcond=None)[0]
        assert kXX_inv_kXx.shape == conditioned_on.shape + points.shape
        res = np.maximum(kxx - (kxX * kXX_inv_kXx.T).sum(axis=1), 0)
        assert res.shape == points.shape
        return res

    if len(conditioned_on) < dimension:
        def density(points: np.ndarray) -> np.ndarray:
            return bayes_kernel_variance(rkhs_kernel, points, conditioned_on)
    else:
        raise NotImplementedError()

    pdf = density(discretisation)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0)
    pdf /= np.sum(pdf)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
        # fig, ax = plt.subplots(1, 1)
        ax.plot(discretisation, pdf / np.trapz(pdf, discretisation))
        for index, point in enumerate(conditioned_on):
            alpha = max(0.5**(len(conditioned_on)-index-1), 0.1)
            ax.axvline(point, color="tab:red", alpha=alpha)
        ax.set_title(f"Optimal sampling distribution at step {len(conditioned_on)+1}")
        plot_path = plot_directory / f"dpp_density_step-{len(conditioned_on)+1}.png"
        print("Saving optimal sampling plot to", plot_path)
        plt.savefig(
            plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
        )
        plt.close(fig)

    return rng.choice(discretisation, p=pdf)
    # return discretisation[np.argmax(pdf)]


if __name__ == "__main__":
    from plotting import plt

    plot_directory = Path(__file__).parent / "plot" / "sampling"
    plot_directory.mkdir(exist_ok=True)

    dimension = 10

    rng = np.random.default_rng(0)
    subspace_basis = orthonormal_basis(hk_gramian(dimension, 1))
    discretisation = np.linspace(-1, 1, 1000)

    sample = []
    for _ in range(2 * dimension):
        sample.append(draw_sample(rng, rkhs_kernel, subspace_basis, dimension, discretisation, plot=True, conditioned_on=sample, regularisation=1e-8))

    sample = []
    for _ in range(dimension):
        sample.append(draw_dpp_sample(rng, rkhs_kernel, dimension, discretisation, plot=True, conditioned_on=sample))

    sample = []
    for _ in range(dimension):
        sample.append(draw_subspace_dpp_sample(rng, subspace_basis, dimension, discretisation, plot=True, conditioned_on=sample))


class Sampler(object):
    def __init__(self, dimension: int, discretisation: np.ndarray) -> None:
        self.discretisation = discretisation
        self.subspace_basis = orthonormal_basis(hk_gramian(dimension, 1))
        self.dimension = dimension

    def draw(self, rng: np.random.Generator, sample_size: int) -> np.ndarray:
        sample = []
        for _ in range(sample_size):
            sample.append(draw_sample(rng, rkhs_kernel, self.subspace_basis, self.dimension, self.discretisation, conditioned_on=sample))
        return np.asarray(sample)


if __name__ == "__main__":
    from tqdm import trange

    from least_squares import quasi_optimality_constant, bias_constant, optimal_least_squares


    sample_size = 20
    # sample_size = 100
    trials = 1_000
    # trials = 10

    oversampling = sample_size / dimension
    assert int(oversampling) == oversampling
    oversampling = int(oversampling)

    plot_directory = Path(__file__).parent / "plot" / f"sampling-{oversampling}"
    plot_directory.mkdir(exist_ok=True)

    l2_legendre = orthonormal_basis(hk_gramian(dimension, 0))
    h1_legendre = orthonormal_basis(hk_gramian(dimension, 1))

    # target = lambda x: np.sin(2 * np.pi * x) + np.cos(2 * dimension * np.pi * x)

    # sampler = Sampler(dimension, discretisation)
    # sample_points = sampler.draw(rng, sample_size)

    # c1 = quasi_optimality_constant(sample_points, dimension)
    # c2 = bias_constant(sample_points, dimension)
    # print(f"Quasi-optimality: {c1:.2f}")
    # print(f"Bias: {c2:.2f}")

    # plot_path = plot_directory / f"error.png"
    # fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    # xs = np.linspace(-1, 1, 1000)
    # c = optimal_least_squares(target, sample_points, dimension, basis=h1_legendre)
    # error = target(xs) - h1_legendre(xs, c)
    # node_errors = target(sample_points) - h1_legendre(sample_points, c)
    # ax.plot(xs, error, "C0-", label="Optimal approximation")
    # ax.plot(sample_points, node_errors, "C0o")
    # c = np.linalg.lstsq(h1_legendre(sample_points, np.eye(dimension)).T, target(sample_points), rcond=None)[0]
    # error = target(xs) - h1_legendre(xs, c)
    # ax.plot(xs, error, "C1--", label="$H^1$ least squares")
    # c = np.linalg.lstsq(l2_legendre(sample_points, np.eye(dimension)).T, target(sample_points), rcond=None)[0]
    # error = target(xs) - l2_legendre(xs, c)
    # ax.plot(xs, error, "C2-.", label="$L^2$ least squares")
    # ax.legend()
    # ax.set_title("Pointwise error")
    # print("Saving error plot to", plot_path)
    # plt.savefig(
    #     plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    # )
    # plt.close(fig)


    def l2_gramian_cond(sample):
        M = l2_legendre(sample, np.eye(dimension)) / np.sqrt(len(sample))
        return np.linalg.cond(M)


    def plot_sample_statistic(samples, plot_path):
        fig, ax = plt.subplots(1, 3, figsize=(8, 4), dpi=300)
        c1s = []
        c2s = []
        c3s = []
        for sample in samples:
            c1s.append(quasi_optimality_constant(sample, dimension, basis=h1_legendre))
            c2s.append(bias_constant(sample, dimension, basis=h1_legendre))
            c3s.append(l2_gramian_cond(sample))
        ax[0].hist(c1s, bins=25)
        ax[0].set_title("Quasi-optimality factor")
        ax[1].hist(c2s, bins=25)
        ax[1].set_title("Noise amplification factor")
        ax[2].hist(c3s, bins=25)
        ax[2].set_title("$L^2$-Gramian condition number")
        # samples = np.concatenate(samples)
        # bins = np.linspace(-1, 1, 80)
        # ax[0].hist(samples, density=True, bins=bins)
        # ax[0].set_title("Sample distribution")
        print("Saving sample statistics plot to", plot_path)
        plt.savefig(
            plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
        )
        plt.close(fig)

    sampler = Sampler(dimension, discretisation)
    samples = []
    for trial in trange(trials):
        samples.append(sampler.draw(rng, sample_size))
    plot_path = plot_directory / "sample_statistics.png"
    plot_sample_statistic(samples, plot_path)

    sampler = Sampler(dimension, discretisation)
    repeated_samples = []
    for trial in trange(trials):
        sample = []
        while len(sample) < sample_size:
            sample = np.concatenate([sample, sampler.draw(rng, dimension)])
        sample = sample[:sample_size]
        repeated_samples.append(sample)
    plot_path = plot_directory / "repeated_sample_statistics.png"
    plot_sample_statistic(repeated_samples, plot_path)

    repeated_dpp_samples = []
    for trial in trange(trials):
        sample = []
        while len(sample) < sample_size:
            partial_sample = []
            for _ in range(dimension):
                partial_sample.append(draw_dpp_sample(rng, rkhs_kernel, dimension, discretisation, conditioned_on=partial_sample))
            sample = np.concatenate([sample, partial_sample])
        sample = sample[:sample_size]
        repeated_dpp_samples.append(sample)
    plot_path = plot_directory / "repeated_dpp_sample_statistics.png"
    plot_sample_statistic(repeated_dpp_samples, plot_path)

    repeated_subspace_dpp_samples = []
    for trial in trange(trials):
        sample = []
        while len(sample) < sample_size:
            partial_sample = []
            for _ in range(dimension):
                # partial_sample.append(draw_subspace_dpp_sample(rng, h1_legendre, dimension, discretisation, conditioned_on=partial_sample))
                partial_sample.append(draw_subspace_dpp_sample(rng, l2_legendre, dimension, discretisation, conditioned_on=partial_sample))
            sample = np.concatenate([sample, partial_sample])
        sample = sample[:sample_size]
        repeated_subspace_dpp_samples.append(sample)
    plot_path = plot_directory / "repeated_subspace_dpp_sample_statistics.png"
    plot_sample_statistic(repeated_subspace_dpp_samples, plot_path)


    plot_path = plot_directory / "compare_sample_statistics.png"
    plt.style.use('seaborn-v0_8-deep')

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
    c1s = lambda samples: [quasi_optimality_constant(sample, dimension, basis=h1_legendre) for sample in samples]
    c1s = np.asarray([c1s(samples), c1s(repeated_samples), c1s(repeated_subspace_dpp_samples), c1s(repeated_dpp_samples)])
    x_min, x_max = np.min(c1s), np.max(c1s[:2])
    x_min -= 0.05 * (x_max - x_min)
    x_max += 0.05 * (x_max - x_min)
    bins = np.linspace(x_min, x_max, 20)
    c1s = np.clip(c1s, x_min, x_max)
    ax[0].hist(list(c1s), bins=bins, density=True, label=["Full kernel", "Repeated kernel", "Repeated subspace-DPP", "Repeated DPP"])
    ax[0].set_title("Quasi-optimality factor")
    ax[0].legend()

    c2s = lambda samples: [bias_constant(sample, dimension, basis=h1_legendre) for sample in samples]
    c2s = np.asarray([c2s(samples), c2s(repeated_samples), c2s(repeated_subspace_dpp_samples), c2s(repeated_dpp_samples)])
    x_min, x_max = np.min(c2s), np.max(c2s[:2])
    x_min -= 0.05 * (x_max - x_min)
    x_max += 0.05 * (x_max - x_min)
    bins = np.linspace(x_min, x_max, 20)
    c2s = np.clip(c2s, x_min, x_max)
    ax[1].hist(list(c2s), bins=bins, density=True, label=["Full kernel", "Repeated kernel", "Repeated subspace-DPP", "Repeated DPP"])
    ax[1].set_title("Noise amplification factor")
    ax[1].legend()

    c3s = lambda samples: [l2_gramian_cond(sample) for sample in samples]
    c3s = np.asarray([c3s(samples), c3s(repeated_samples), c3s(repeated_subspace_dpp_samples), c3s(repeated_dpp_samples)])
    x_min, x_max = np.min(c3s), np.max(c3s[:2])
    x_min -= 0.05 * (x_max - x_min)
    x_max += 0.05 * (x_max - x_min)
    bins = np.linspace(x_min, x_max, 20)
    c3s = np.clip(c3s, x_min, x_max)
    ax[2].hist(list(c3s), bins=bins, density=True, label=["Full kernel", "Repeated kernel", "Repeated subspace-DPP", "Repeated DPP"])
    ax[2].set_title("$L^2$-Gramian condition number")
    ax[2].legend()

    print("Saving sample statistics plot to", plot_path)
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )
    plt.close(fig)
