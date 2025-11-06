from typing import Optional
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import trange

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


integrals = None


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

    global integrals
    if integrals is None:
        integrals = [[] for _ in range(subspace_basis.dimension)]

    subspace_kernel_l2 = create_subspace_kernel(l2_basis)
    if len(conditioned_on) < subspace_basis.dimension:
        subspace_kernel = create_subspace_kernel(subspace_basis)

        # Check if the integrals of the conditional densities are independen of the sample on which is conditioned.
        xs = np.linspace(*subspace_basis.domain, 1000)
        assert subspace_basis is rkhs_basis
        assert regularisation == 1e-8
        integrand = bayes_kernel_variance(subspace_kernel, xs, conditioned_on)
        integrand /= bayes_kernel_variance(rkhs_kernel, xs, conditioned_on) + regularisation

        Z1 = np.trapezoid(subspace_kernel_l2(xs, xs) * rho(xs), xs)
        assert abs(Z1 - l2_basis.dimension) < 1e-3
        Z2 = np.trapezoid(rkhs_kernel(xs, xs) * rho(xs), xs)

        reference_density = rkhs_kernel(xs, xs) / subspace_kernel(xs, xs) * rho(xs)
        reference_density *= 0.5 * (subspace_kernel_l2(xs, xs) / Z1 + subspace_kernel(xs, xs) / Z2)

        integral = np.trapezoid(integrand * reference_density, xs)
        integrals[len(conditioned_on)].append(integral)
        # vrs_integral = l2_basis.dimension - len(conditioned_on)
        # print(f"  Sample size: {len(conditioned_on)}  |  Integral: {integral:.2e}")
        # print(f"  Sample size: {len(conditioned_on)}  |  Integral: {integrals[len(conditioned_on)][-1]:.2e}  |  VRS integral: {vrs_integral}  |  Error: {integrals[len(conditioned_on)][-1] - vrs_integral:.2e}")
        # if len(integrals[0]) > 10:
        #     exit()
        # assert vrs_integral - 1 < integral <= vrs_integral + 0.5, f"Integral: {integral:.2e}  |  VRS integral: {vrs_integral}  |  Error: {integral - vrs_integral:.2e}"

        def density(points: np.ndarray) -> np.ndarray:
            numerator = bayes_kernel_variance(subspace_kernel, points, conditioned_on)
            denominator = bayes_kernel_variance(rkhs_kernel, points, conditioned_on) + regularisation

            reference_density = rkhs_kernel(points, points) / subspace_kernel(points, points) * rho(points)
            reference_density *= 0.5 * (subspace_kernel_l2(points, points) / Z1 + subspace_kernel(points, points) / Z2)

            return numerator / denominator * reference_density
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

            reference_density = subspace_kernel_l2(points, points) / l2_basis.dimension * rho(points)

            return (1 + numerator / denominator) * reference_density

    if plot:
        plot_path = plot_directory / f"embedding_density_step-{len(conditioned_on)+1}.png"
        __plot_step(plot_path, density, discretisation, conditioned_on)

    return draw_sample(rng, density, discretisation)


def draw_volume_sample(
        rng: np.random.Generator,
        rkhs_kernel: Kernel,
        reference_density: Density,
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
        return bayes_kernel_variance(rkhs_kernel, points, conditioned_on) * reference_density(points)

    if plot:
        plot_path = plot_directory / f"volume_density_step-{len(conditioned_on)+1}.png"
        __plot_step(plot_path, density, discretisation, conditioned_on)

    return draw_sample(rng, density, discretisation)


def draw_subspace_volume_sample(
        rng: np.random.Generator,
        subspace_basis: Basis,
        reference_density: Density,
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
            return bayes_kernel_variance(subspace_kernel, points, conditioned_on) * reference_density(points)
    else:
        christoffel_density = compute_subspace_volume_marginal(subspace_basis)

        def density(points: np.ndarray) -> np.ndarray:
            return christoffel_density(points) * reference_density(points)


    if plot:
        plot_path = plot_directory / f"subspace_volume_density_step-{len(conditioned_on)+1}.png"
        __plot_step(plot_path, density, discretisation, conditioned_on)

    return draw_sample(rng, density, discretisation)


Sampler = Callable[[list[float]], float]


def draw_sequence(sampler: Sampler, sample_size: int, *, initial_sample: Optional[list[float]] = None, verbose: bool = False) -> np.ndarray:
    if initial_sample is None:
        sample = []
    else:
        sample = list(initial_sample)
    _range = trange if verbose else range
    for _ in _range(sample_size):
        sample.append(sampler(conditioned_on=sample))
    return sample


def kernel_gramian(kernel: Kernel, basis: Basis, points: np.ndarray, *, regularisation: float = 1e-8) -> np.ndarray:
    points = np.asarray(points)
    assert points.ndim == 1
    K = kernel_matrix(kernel, points)
    assert K.shape == (len(points), len(points))
    M = basis(points)
    assert M.shape == (basis.dimension, len(points))
    es, vs = np.linalg.eigh(K)
    assert np.max(abs(vs * es @ vs.T - K)) <= 1e-12 * np.max(abs(K))
    es = np.maximum(es, regularisation)
    M = M @ vs
    return M / es @ M.T


def quasi_optimality_constants(kernel: Kernel, basis: Basis, points: np.ndarray) -> tuple[float, float, float]:
    G = kernel_gramian(kernel, basis, points)
    assert G.shape == (basis.dimension, basis.dimension)
    I = np.eye(basis.dimension)
    mu = 1 / np.sqrt(np.linalg.norm(G, ord=-2))
    tau = min(np.linalg.norm(I - G, ord="fro"), 1.0)
    assert np.isfinite(mu) and np.isfinite(tau)
    return mu, tau, (1 + (mu * tau)**2)


if __name__ == "__main__":
    from functools import partial

    import matplotlib.pyplot as plt

    from rkhs_1d import kernel_matrix, H10Kernel, H1Kernel, H1GaussKernel
    from basis_1d import compute_discrete_gramian, enforce_zero_trace, orthonormalise, MonomialBasis, FourierBasis, SinBasis

    rng = np.random.default_rng(0)

    # dimension = 5
    dimension = 10
    basis_name = "polynomial"
    # basis_name = "fourier"

    # oversampling = 1
    # oversampling = 2
    oversampling = 10
    sample_size = oversampling * dimension

    # trials = 10
    # trials = 100
    trials = 1_000

    spaces = ["h10", "h1", "h1gauss"]

    PAPER_Z_INTEGRAL_PLOT = True
    if PAPER_Z_INTEGRAL_PLOT:
        dimension = 10
        basis_name = "polynomial"
        oversampling = 10
        sample_size = oversampling * dimension
        trials = 100
        spaces = ["h10"]

    for space in spaces:
        print(f"Compute sample statistics for {space} with {basis_name} basis")
        if space == "h10":
            rkhs_kernel = H10Kernel((-1, 1))
            if basis_name == "polynomial":
                initial_basis = MonomialBasis(dimension+2, domain=(-1, 1))
            elif basis_name == "fourier":
                initial_basis = SinBasis(dimension+2, domain=(-1, 1))
            initial_basis = enforce_zero_trace(initial_basis)
        elif space == "h1":
            rkhs_kernel = H1Kernel((-1, 1))
            if basis_name == "polynomial":
                initial_basis = MonomialBasis(dimension, domain=(-1, 1))
            elif basis_name == "fourier":
                initial_basis = FourierBasis(dimension, domain=(-1, 1))
        elif space == "h1gauss":
            # rkhs_kernel = H1GaussKernel((-5, 5))
            rkhs_kernel = H1GaussKernel((-8, 8))
            if basis_name == "polynomial":
                # basisval = MonomialBasis(dimension, domain=(-5, 5))
                initial_basis = MonomialBasis(dimension, domain=(-8, 8))
            elif basis_name == "fourier":
                # basisval = FourierBasis(dimension, domain=(-5, 5))
                initial_basis = FourierBasis(dimension, domain=(-8, 8))
        else:
            raise NotImplementedError()

        discretisation = np.linspace(*initial_basis.domain, 1000)

        if space == "h1gauss":
            discrete_l2_gramian = compute_discrete_gramian("l2gauss", initial_basis.domain, 2 ** 13)
        else:
            discrete_l2_gramian = compute_discrete_gramian("l2", initial_basis.domain, 2 ** 13)
        l2_basis = orthonormalise(initial_basis, *discrete_l2_gramian)

        discrete_rkhs_gramian = compute_discrete_gramian(space, initial_basis.domain, 2 ** 13)
        rkhs_basis = orthonormalise(initial_basis, *discrete_rkhs_gramian)

        # for _ in range(2 * dimension):
        #     draw_embedding_sample(rng, h1_kernel, h1_basis, discretisation, plot=True, conditioned_on=sample, regularisation=1e-8)

        # for _ in range(dimension):
        #     draw_volume_sample(rng, h1_kernel, discretisation, plot=True, conditioned_on=sample)
        #     draw_subspace_volume_sample(rng, h1_basis, discretisation, plot=True, conditioned_on=sample)

        if space in ["h10", "h1"]:
            def rho(x):
                return np.full(len(x), 0.5)

        elif space == "h1gauss":
            def rho(x):
                return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

        embedding_sampler = partial(draw_embedding_sample, rng=rng, rkhs_kernel=rkhs_kernel, subspace_basis=rkhs_basis, discretisation=discretisation)
        volume_sampler = partial(draw_volume_sample, rng=rng, rkhs_kernel=rkhs_kernel, reference_density=rho, discretisation=discretisation)
        subspace_volume_sampler = partial(draw_subspace_volume_sample, rng=rng, subspace_basis=l2_basis, reference_density=rho, discretisation=discretisation)

        # def marginal_embedding_sampler(conditioned_on: Optional[list[float]] = None) -> float:
        #     TODO: weighting...
        #     return draw_sample(rng=rng, density=compute_embedding_marginal(rkhs_kernel, rkhs_basis), discretisation=discretisation)

        def marginal_subspace_volume_sampler(conditioned_on: Optional[list[float]] = None) -> float:
            density = compute_subspace_volume_marginal(l2_basis)
            weighted_density = lambda x: density(x) * rho(x)
            return draw_sample(rng=rng, density=weighted_density, discretisation=discretisation)

        plot_directory = Path(__file__).parent / "plot"
        plot_directory.mkdir(exist_ok=True)

        # samplers = [embedding_sampler, volume_sampler, subspace_volume_sampler, marginal_embedding_sampler, marginal_subspace_volume_sampler]
        # sampler_names = ["Embedding sampler", "Volume sampler", "Subspace volume sampler", "Marginal embedding sampler", "Marginal subspace volume sampler"]
        samplers = [embedding_sampler, volume_sampler, subspace_volume_sampler, marginal_subspace_volume_sampler]
        sampler_names = ["Embedding sampling", "Volume sampling", "Subspace volume sampling", "Christoffel sampling"]
        if PAPER_Z_INTEGRAL_PLOT:
            samplers = [embedding_sampler]
            sampler_names = ["Embedding sampling"]
        constants = np.empty((len(samplers), trials, 3))
        for index in range(len(samplers)):
            print(f"Sampling scheme: {sampler_names[index]}")
            if sampler_names[index] == "Embedding sampling":
                integrals = None
            for trial in trange(trials):
                sample = draw_sequence(samplers[index], sample_size)
                constants[index, trial] = quasi_optimality_constants(rkhs_kernel, rkhs_basis, sample)
            if sampler_names[index] == "Embedding sampling":
                file_path = sampler_names[index].replace(" ", "_").lower()
                file_path = plot_directory / f"integrals_{space}_{file_path}.npy"
                print(f"Saving integrals to '{file_path}'")
                np.save(file_path, integrals)
                if PAPER_Z_INTEGRAL_PLOT:
                    exit()

        plt.style.use('seaborn-v0_8-deep')
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        for index, statistic in enumerate([r"$\mu$", r"$\tau$", r"$1 + \mu^2\tau^2$"]):
            values = constants[..., index]
            assert values.shape == (len(samplers), trials)
            x_min = [1, 0, 1][index]
            if index == 1:
                x_max = 1
            else:
                x_max = np.quantile(values[0], 0.9)
                x_max += 0.05 * (x_max - x_min)
            bins = np.linspace(x_min, x_max, 20)
            values = np.clip(values, x_min, x_max)
            ax[index].hist(list(values), bins=bins, density=True, label=sampler_names)
            # ax[index].hist(list(c1s), bins=bins, density=True, histtype='step', stacked=True, fill=False, label=sampler_names)
            ax[index].set_title(statistic)
            ax[index].legend(fontsize=8)

        plot_path = plot_directory / f"sample_statistics_{space}_{basis_name}_{oversampling}x.png"
        print("Saving sample statistics plot to", plot_path)
        plt.savefig(
            plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
        )
        plt.close(fig)
