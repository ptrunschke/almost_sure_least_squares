from typing import Optional
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import trange

from basis_1d import Basis, create_subspace_kernel
from rkhs_1d import Kernel


Density = Callable[[np.ndarray], np.ndarray]
Sampler = Callable[[list[float]], float]


# REGULARISATION = 1e-8
REGULARISATION = 1e-12


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


def draw_sample(rng: np.random.Generator, density: Density, discretisation: np.ndarray) -> float:
    pdf = density(discretisation)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0)
    pdf /= np.sum(pdf)
    return rng.choice(discretisation, p=pdf)
    # return discretisation[np.argmax(pdf)]


def draw_embedding_sample(
        rng: np.random.Generator,
        rkhs_kernel: Kernel,
        rkhs_basis: Basis,
        l2_basis: Basis,
        reference_density: Density,
        discretisation: np.ndarray,
        *,
        conditioned_on: Optional[list[float]] = None,
        regularisation: float = REGULARISATION,
    ) -> float:
    if conditioned_on is None:
        conditioned_on = []
    discretisation, conditioned_on = np.asarray(discretisation), np.asarray(conditioned_on)
    assert discretisation.ndim == 1 and conditioned_on.ndim == 1

    # subspace_l2_kernel = create_subspace_kernel(l2_basis)

    if len(conditioned_on) < rkhs_basis.dimension:
        subspace_kernel = create_subspace_kernel(rkhs_basis)

        def density(points: np.ndarray) -> np.ndarray:
            numerator = bayes_kernel_variance(subspace_kernel, points, conditioned_on)
            denominator = bayes_kernel_variance(rkhs_kernel, points, conditioned_on) + regularisation
            density = numerator / denominator
            # density *= np.nan_to_num(subspace_l2_kernel(points, points) / subspace_kernel(points, points))
            density *= rkhs_kernel(points, points) * reference_density(points)
            return density
    else:
        def density(points: np.ndarray) -> np.ndarray:
            bc = rkhs_basis(conditioned_on)
            dimension = bc.shape[0]
            assert bc.shape == (rkhs_basis.dimension,) + conditioned_on.shape
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
            bx = rkhs_basis(points)
            assert bx.shape == (dimension,) + points.shape
            diff = bx - bckx
            assert diff.shape == (dimension,) + points.shape
            numerator = (diff * np.linalg.lstsq(G, diff, rcond=None)[0]).sum(axis=0)
            assert numerator.shape == points.shape
            numerator = np.maximum(numerator, 0)
            denominator = bayes_kernel_variance(rkhs_kernel, points, conditioned_on) + regularisation
            density = 1 + numerator / denominator
            # density *= subspace_l2_kernel(points, points) * reference_density(points)
            density *= rkhs_kernel(points, points) * reference_density(points)
            return density

    return draw_sample(rng, density, discretisation)


def draw_volume_sample(
        rng: np.random.Generator,
        rkhs_kernel: Kernel,
        reference_density: Density,
        discretisation: np.ndarray,
        *,
        conditioned_on: Optional[list[float]] = None,
    ) -> float:
    if conditioned_on is None:
        conditioned_on = []
    discretisation, conditioned_on = np.asarray(discretisation), np.asarray(conditioned_on)
    assert discretisation.ndim == 1 and conditioned_on.ndim == 1

    def density(points: np.ndarray) -> np.ndarray:
        return bayes_kernel_variance(rkhs_kernel, points, conditioned_on) * reference_density(points)

    return draw_sample(rng, density, discretisation)


def draw_sequence(sampler: Sampler, sample_size: int, *, initial_sample: Optional[list[float]] = None, verbose: bool = False) -> np.ndarray:
    if initial_sample is None:
        sample = []
    else:
        sample = list(initial_sample)
    _range = trange if verbose else range
    for _ in _range(sample_size):
        sample.append(sampler(conditioned_on=sample))
    return sample


def kernel_gramian(kernel: Kernel, basis: Basis, points: np.ndarray, *, regularisation: float = REGULARISATION) -> np.ndarray:
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


def quasi_optimality_constant(kernel: Kernel, basis: Basis, points: np.ndarray) -> float:
    G = kernel_gramian(kernel, basis, points)
    assert G.shape == (basis.dimension, basis.dimension)
    mu_inv = np.sqrt(np.linalg.norm(G, ord=-2))
    assert np.isfinite(mu_inv), mu_inv
    return 1 / mu_inv


if __name__ == "__main__":
    from functools import partial

    from tqdm import tqdm
    import ray

    from rkhs_1d import kernel_matrix, H10Kernel, H1Kernel, H1GaussKernel
    from basis_1d import compute_discrete_gramian, enforce_zero_trace, orthonormalise, MonomialBasis, FourierBasis, SinBasis

    rng = np.random.default_rng(0)

    basis_name = "polynomial"
    # basis_name = "fourier"

    trials = 200
    SHORTCUT = False
    # NOTE: This shortcut is justified as follows. If we would reuse the old sample points in every step, and only add new ones,
    #       the monotonicity of mu would ensure that, once mu <= 2 for n samples, mu <= 2 for n+k samples for all k >= 0.

    def setup(space, basis, dimension):
        if space == "h10":
            rkhs_kernel = H10Kernel((-1, 1))
            if basis == "polynomial":
                initial_basis = MonomialBasis(dimension+2, domain=(-1, 1))
            elif basis == "fourier":
                initial_basis = SinBasis(dimension+2, domain=(-1, 1))
            initial_basis = enforce_zero_trace(initial_basis)
        elif space == "h1":
            rkhs_kernel = H1Kernel((-1, 1))
            if basis == "polynomial":
                initial_basis = MonomialBasis(dimension, domain=(-1, 1))
            elif basis == "fourier":
                initial_basis = FourierBasis(dimension, domain=(-1, 1))
        elif space == "h1gauss":
            rkhs_kernel = H1GaussKernel((discretisation[0], discretisation[-1]))
            if basis == "polynomial":
                initial_basis = MonomialBasis(dimension, domain=(discretisation[0], discretisation[-1]))
            elif basis == "fourier":
                initial_basis = FourierBasis(dimension, domain=(discretisation[0], discretisation[-1]))
        else:
            raise NotImplementedError()

        if space == "h1gauss":
            discrete_l2_gramian = compute_discrete_gramian("l2gauss", initial_basis.domain, 2 ** 13)
        else:
            discrete_l2_gramian = compute_discrete_gramian("l2", initial_basis.domain, 2 ** 13)
        l2_basis = orthonormalise(initial_basis, *discrete_l2_gramian)

        discrete_rkhs_gramian = compute_discrete_gramian(space, initial_basis.domain, 2 ** 13)
        rkhs_basis = orthonormalise(initial_basis, *discrete_rkhs_gramian)

        return rkhs_kernel, l2_basis, rkhs_basis


    plot_directory = Path(__file__).parent / "new_plots"
    plot_directory.mkdir(exist_ok=True)
    for space in ["h10", "h1", "h1gauss"]:
        print(f"Space: {space}")
        print(f"Basis: {basis_name}")
        if space in ["h10", "h1"]:
            # dimensions = np.arange(0, 40, 1) + 1
            # sample_sizes = np.arange(0, 140, 2) + 1
            dimensions = np.arange(0, 25, 1) + 1
            sample_sizes = np.arange(0, 80, 2) + 1
            discretisation = np.linspace(-1, 1, 1000)
            rho = lambda x: np.full(len(x), 0.5)
        elif space == "h1gauss":
            dimensions = np.arange(0, 15, 1) + 1
            sample_sizes = np.arange(0, 48, 2) + 1  # len(sample_sizes of h1) / len(dimensions of h1) * len(dimensions of h1gauss) == 24 == 48 / 2
            discretisation = np.linspace(-16, 16, 1000)
            rho = lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

        for sampler_name in ["Christoffel sampling", "Volume sampling", "Embedding sampling"]:
            print(f"Sampling scheme: {sampler_name}")

            constants = np.empty((len(dimensions), len(sample_sizes), trials))
            for j, dimension in tqdm(enumerate(dimensions), desc="Dimension", total=len(dimensions), position=0):
                rkhs_kernel, l2_basis, rkhs_basis = setup(space, basis_name, dimension)

                # subspace_l2_kernel = create_subspace_kernel(l2_basis)
                subspace_kernel = create_subspace_kernel(rkhs_basis)
                def christoffel_sampler(conditioned_on: Optional[list[float]] = None) -> float:
                    # weighted_density = lambda x: subspace_l2_kernel(x, x) * rho(x)
                    weighted_density = lambda x: subspace_kernel(x, x) * rho(x)
                    return draw_sample(rng=rng, density=weighted_density, discretisation=discretisation)

                if sampler_name == "Embedding sampling":
                    sampler = partial(draw_embedding_sample, rng=rng, rkhs_kernel=rkhs_kernel, rkhs_basis=rkhs_basis, l2_basis=l2_basis, reference_density=rho, discretisation=discretisation)
                elif sampler_name == "Volume sampling":
                    sampler = partial(draw_volume_sample, rng=rng, rkhs_kernel=rkhs_kernel, reference_density=rho, discretisation=discretisation)
                elif sampler_name == "Christoffel sampling":
                    sampler = christoffel_sampler
                else:
                    raise NotImplementedError(f"Unknown sampling method: '{sampler_name}'")
                
                @ray.remote
                def draw_trial(sample_size: int, trial: int) -> float:
                    sample = draw_sequence(sampler, sample_size)
                    return quasi_optimality_constant(rkhs_kernel, rkhs_basis, sample)

                for k, sample_size in tqdm(enumerate(sample_sizes), desc="Sample size", total=len(sample_sizes), position=1, leave=False):
                    futures = [draw_trial.remote(sample_size, trial) for trial in range(trials)]
                    constants[j, k, :] = ray.get(futures)
                    if SHORTCUT and np.all(constants[j, k] <= 2):
                        constants[j, k+1:] = 0
                        break

            file_path = sampler_name.replace(" ", "_").lower()
            file_path = plot_directory / f"suboptimality_constants_{space}_{file_path}.npz"
            np.savez_compressed(file_path, ds=dimensions, ss=sample_sizes, mus=constants)
