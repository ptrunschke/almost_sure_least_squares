from typing import Optional
from collections.abc import Callable
from pathlib import Path

import numpy as np


from basis_1d import Basis, create_subspace_kernel
from rkhs_1d import Kernel


def bayes_kernel_variance(kernel: Kernel, points: np.ndarray, conditioned_on: np.ndarray) -> np.ndarray:
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


def compute_embedding_marginal(kernel: Kernel, basis: Basis):
    subspace_kernel = create_subspace_kernel(basis)
    def density(points: np.ndarray) -> np.ndarray:
        return subspace_kernel(points, points) / kernel(points, points)
    return density


def compute_subspace_volume_marginal(basis: Basis):
    subspace_kernel = create_subspace_kernel(basis)
    def density(points: np.ndarray) -> np.ndarray:
        return subspace_kernel(points, points)
        # return np.sum(basis(points)**2, axis=0) / basis.dimension
    return density


Density = Callable[[np.ndarray], np.ndarray]


def draw_sample(rng: np.random.Generator, density: Density, discretisation: np.ndarray) -> float:
    pdf = density(discretisation)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0)
    pdf /= np.sum(pdf)
    return rng.choice(discretisation, p=pdf)
    # return discretisation[np.argmax(pdf)]


def __plot_step(plot_path: Path, density: Density, discretisation: np.ndarray, conditioned_on: list[float]):
    pdf = density(discretisation)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0)
    pdf /= np.trapz(pdf, discretisation)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    ax.plot(discretisation, pdf)
    for index, point in enumerate(conditioned_on):
        alpha = max(0.5**(len(conditioned_on)-index-1), 0.1)
        ax.axvline(point, color="tab:red", alpha=alpha)
    ax.set_title(f"Step {len(conditioned_on)+1}")
    print("Saving optimal sampling plot to", plot_path)
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )
    plt.close(fig)


def draw_embedding_sample(
        rng: np.random.Generator,
        rkhs_kernel: Kernel,
        subspace_basis: Basis,
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

    if len(conditioned_on) < subspace_basis.dimension:
        subspace_kernel = create_subspace_kernel(subspace_basis)

        def density(points: np.ndarray) -> np.ndarray:
            numerator = bayes_kernel_variance(subspace_kernel, points, conditioned_on)
            denominator = bayes_kernel_variance(rkhs_kernel, points, conditioned_on) + regularisation
            return numerator / denominator
    else:
        def density(points: np.ndarray) -> np.ndarray:
            bc = subspace_basis(conditioned_on)
            dimension = bc.shape[0]
            assert bc.shape == (subspace_basis.dimension,) + conditioned_on.shape
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
            bx = subspace_basis(points)
            assert bx.shape == (dimension,) + points.shape
            diff = bx - bckx
            assert diff.shape == (dimension,) + points.shape
            numerator = (diff * np.linalg.lstsq(G, diff, rcond=None)[0]).sum(axis=0)
            assert numerator.shape == points.shape
            numerator = np.maximum(numerator, 0)
            denominator = bayes_kernel_variance(rkhs_kernel, points, conditioned_on) + regularisation
            return 1 + numerator / denominator

    if plot:
        plot_path = plot_directory / f"embedding_density_step-{len(conditioned_on)+1}.png"
        __plot_step(plot_path, density, discretisation, conditioned_on)

    return draw_sample(rng, density, discretisation)


def draw_volume_sample(
        rng: np.random.Generator,
        rkhs_kernel: Kernel,
        discretisation: np.ndarray,
        *,
        plot: bool = False,
        conditioned_on: Optional[list[float]] = None,
    ) -> list[int]:
    if conditioned_on is None:
        conditioned_on = []
    discretisation, conditioned_on = np.asarray(discretisation), np.asarray(conditioned_on)
    assert discretisation.ndim == 1 and conditioned_on.ndim == 1

    def density(points: np.ndarray) -> np.ndarray:
        return bayes_kernel_variance(rkhs_kernel, points, conditioned_on)

    if plot:
        plot_path = plot_directory / f"volume_density_step-{len(conditioned_on)+1}.png"
        __plot_step(plot_path, density, discretisation, conditioned_on)

    return draw_sample(rng, density, discretisation)


def draw_subspace_volume_sample(
        rng: np.random.Generator,
        subspace_basis: Basis,
        discretisation: np.ndarray,
        *,
        plot: bool = False,
        conditioned_on: Optional[list[float]] = None,
    ) -> list[int]:
    if conditioned_on is None:
        conditioned_on = []
    discretisation, conditioned_on = np.asarray(discretisation), np.asarray(conditioned_on)
    assert discretisation.ndim == 1 and conditioned_on.ndim == 1

    if len(conditioned_on) < subspace_basis.dimension:
        subspace_kernel = create_subspace_kernel(subspace_basis)

        def density(points: np.ndarray) -> np.ndarray:
            return bayes_kernel_variance(subspace_kernel, points, conditioned_on)
    else:
        density = compute_subspace_volume_marginal(subspace_basis)

    if plot:
        plot_path = plot_directory / f"subspace_volume_density_step-{len(conditioned_on)+1}.png"
        __plot_step(plot_path, density, discretisation, conditioned_on)

    return draw_sample(rng, density, discretisation)


Sampler = Callable[[list[float]], float]


def draw_sequence(sampler: Sampler, sample_size: int) -> np.ndarray:
    sample = []
    for _ in range(sample_size):
        sample.append(sampler(conditioned_on=sample))
    return sample



if __name__ == "__main__":
    from functools import partial

    import matplotlib.pyplot as plt
    from tqdm import trange

    from rkhs_1d import H1Kernel, kernel_matrix
    from basis_1d import MonomialBasis, compute_discrete_gramian, orthonormalise

    rng = np.random.default_rng(0)

    h1_kernel = H1Kernel((-1, 1))

    dimension = 10
    basis = MonomialBasis(dimension, domain=(-1, 1))
    discrete_l2_gramian = compute_discrete_gramian("l2", basis.domain, 2 ** 12)
    l2_basis = orthonormalise(basis, *discrete_l2_gramian)
    discrete_h1_gramian = compute_discrete_gramian("h1", basis.domain, 2 ** 12)
    h1_basis = orthonormalise(basis, *discrete_h1_gramian)

    discretisation = np.linspace(-1, 1, 1000)

    # for _ in range(2 * dimension):
    #     draw_embedding_sample(rng, h1_kernel, h1_basis, discretisation, plot=True, conditioned_on=sample, regularisation=1e-8)

    # for _ in range(dimension):
    #     draw_volume_sample(rng, h1_kernel, discretisation, plot=True, conditioned_on=sample)
    #     draw_subspace_volume_sample(rng, h1_basis, discretisation, plot=True, conditioned_on=sample)

    def kernel_gramian(kernel: Kernel, basis: Basis, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        assert points.ndim == 1
        K = kernel_matrix(kernel, points)
        assert K.shape == (len(points), len(points))
        M = basis(points)
        assert M.shape == (basis.dimension, len(points))
        es, vs = np.linalg.eigh(K)
        assert np.allclose(vs * es @ vs.T, K)
        M = M @ vs
        return M / es @ M.T

    def quasi_optimality_constants(kernel: Kernel, basis: Basis, points: np.ndarray) -> tuple[float, float, float]:
        G = kernel_gramian(kernel, basis, points)
        assert G.shape == (basis.dimension, basis.dimension)
        I = np.eye(basis.dimension)
        mu = 1 / np.sqrt(np.linalg.norm(G, ord=-2))
        tau = min(np.linalg.norm(I - G, ord="fro"), 1.0)
        return mu, tau, (1 + (mu * tau)**2)

    sample_size = 10
    # sample_size = 20
    # sample_size = 100
    # trials = 10
    # trials = 100
    trials = 1_000

    oversampling = sample_size / dimension
    assert int(oversampling) == oversampling
    oversampling = int(oversampling)

    embedding_sampler = partial(draw_embedding_sample, rng=rng, rkhs_kernel=h1_kernel, subspace_basis=h1_basis, discretisation=discretisation)
    volume_sampler = partial(draw_volume_sample, rng=rng, rkhs_kernel=h1_kernel, discretisation=discretisation)
    subspace_volume_sampler = partial(draw_subspace_volume_sample, rng=rng, subspace_basis=l2_basis, discretisation=discretisation)

    def marginal_embedding_sampler(conditioned_on: Optional[list[float]] = None) -> float:
        return draw_sample(rng=rng, density=compute_embedding_marginal(h1_kernel, h1_basis), discretisation=discretisation)

    def marginal_subspace_volume_sampler(conditioned_on: Optional[list[float]] = None) -> float:
        return draw_sample(rng=rng, density=compute_subspace_volume_marginal(l2_basis), discretisation=discretisation)

    # samplers = [embedding_sampler, volume_sampler, subspace_volume_sampler, marginal_embedding_sampler, marginal_subspace_volume_sampler]
    # sampler_names = ["Embedding sampler", "Volume sampler", "Subspace volume sampler", "Marginal embedding sampler", "Marginal subspace volume sampler"]
    samplers = [embedding_sampler, volume_sampler, subspace_volume_sampler, marginal_subspace_volume_sampler]
    sampler_names = ["Embedding sampling", "Volume sampling", "Subspace volume sampling", "Christoffel sampling"]
    constants = np.empty((len(samplers), trials, 3))
    for index in range(len(samplers)):
        print(f"Sampling scheme: {sampler_names[index]}")
        for trial in trange(trials):
            sample = draw_sequence(samplers[index], sample_size)
            constants[index, trial] = quasi_optimality_constants(h1_kernel, h1_basis, sample)


    plot_directory = Path(__file__).parent / "plot"
    plot_directory.mkdir(exist_ok=True)
    plot_path = plot_directory / f"sample_statistics_{oversampling}x.png"

    plt.style.use('seaborn-v0_8-deep')
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
    if oversampling == 1:
        bounds = [(1, 20), (0, 1), (1, 100)]
    elif oversampling == 2:
        bounds = [(1, 10), (0, 1), (1, 50)]
    elif oversampling == 10:
        bounds = [(1, 2.5), (0, 1), (1, 5)]
    else:
        raise NotImplementedError()
    for index, statistic in enumerate([r"$\mu$", r"$\tau$", r"$1 + \mu^2\tau^2$"]):
        values = constants[..., index]
        assert values.shape == (len(samplers), trials)
        # x_min, x_max = np.min(values), np.quantile(values, 0.9)
        # x_min -= 0.05 * (x_max - x_min)
        # x_max += 0.05 * (x_max - x_min)
        x_min, x_max = bounds[index]
        bins = np.linspace(x_min, x_max, 20)
        values = np.clip(values, x_min, x_max)
        ax[index].hist(list(values), bins=bins, density=True, label=sampler_names)
        # ax[index].hist(list(c1s), bins=bins, density=True, histtype='step', stacked=True, fill=False, label=sampler_names)
        ax[index].set_title(statistic)
        ax[index].legend(fontsize=8)
    print("Saving sample statistics plot to", plot_path)
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )
    plt.close(fig)
