from __future__ import annotations
from typing import Optional
from jaxtyping import Float
import numpy as np
from opt_einsum import contract
# from rkhs_1d import Kernel
from basis_1d import TransformedBasis, Basis
from sampling import draw_weighted_sequence, Sampler
from greedy_subsampling import fast_greedy_step, Metric, FastMetric
from tensor_train import TensorTrain, TensorTrainCoreSpace


# def ensure_stability(core_basis: CoreBasis, target_metric: Metric, target: float, repetitions: int = 10) -> np.ndarray:
#     # Draw new samples until the target is over-satisfied by a factor of 1/(1-1/e) = e / (e - 1).
#     dimension = np.reshape(core_basis.domain, (-1, 2)).shape[0]
#     print("Ensuring stability...")
#     core_space_sampler = create_core_space_sampler(rng, core_basis, discretisation)
#     candidates = []
#     target = np.e / (np.e - 1) * target
#     value = -np.inf
#     while value < target:
#         for _ in range(repetitions):
#             extension = draw_weighted_sequence(core_space_sampler, core_basis.dimension)[0]
#             extended_value = target_metric(candidates + extension)
#             if extended_value > value:
#                 extended_candidates = candidates + extension
#                 value = extended_value
#         candidates = extended_candidates
#         print(f"  Sample size: {len(candidates)}  |  Target metric: {value:.2e} < {target:.2e}")
#     candidates = np.asarray(candidates)
#     assert candidates.ndim == 2 and candidates.shape[1] == dimension
#     return candidates


# def greedy_bound(
#     selection_metric_factory: FastMetric,
#     target_metric: Metric,
#     target: float,
#     full_sample: np.ndarray,
#     selected: Optional[list[bool]] = None,
# ) -> list[bool]:
#     print("Subsampling...")
#     if selected is None:
#         selected = np.full(len(full_sample), False, dtype=bool)
#     else:
#         selected = np.array(selected)
#     indices = np.arange(len(full_sample))
#     value = -np.inf
#     for selection_size in range(np.count_nonzero(selected), len(full_sample)):
#         if value >= target:
#             break
#         selection_metric = selection_metric_factory(full_sample[selected])
#         selected[indices[~selected][fast_greedy_step(selection_metric, full_sample[~selected])]] = True
#         value = target_metric(full_sample[selected])
#         print(f"  Sample size: {selection_size}  |  Target metric: {value:.2e} < {target:.2e}")
#     return selected


# def greedy_draw(
#     selection_metric_factory: FastMetric,
#     selection_size: int,
#     full_sample: np.ndarray,
#     selected: Optional[list[bool]] = None,
# ) -> list[bool]:
#     print("Subsampling...")
#     if selected is None:
#         selected = np.full(len(full_sample), False, dtype=bool)
#     else:
#         selected = np.array(selected)
#     indices = np.arange(len(full_sample))
#     for _ in range(selection_size - np.count_nonzero(selected)):
#         selection_metric = selection_metric_factory(full_sample[selected])
#         selected[indices[~selected][fast_greedy_step(selection_metric, full_sample[~selected])]] = True
#     return selected


def fast_eta_l2_metric(
    basis: Basis, samples: Float[np.ndarray, "sample_size dimension"], weights: Float[np.ndarray, " sample_size"]
) -> FastMetric:
    # dimension = np.reshape(basis.domain, (-1, 2)).shape[0]
    # We want to compute eta_l2(x) = sum_j bs[j] @ bs[j], where bs[j] = basis(x)[j].
    # But note that G(x + [y]) = n/(n+1) G(x) + 1/(n+1) G(y) for all sample points y.
    # So, maximising trace(G(x + [y])) is equivalent to just maximising trace(G([y])).
    B = basis(samples)
    trG = np.sum(B**2 @ weights) / len(samples)

    # B_norms = B**2 @ weights
    # assert B_norms.shape == (basis.dimension,)
    def eta_l2(new_points: np.ndarray, new_weights: np.ndarray) -> np.ndarray:
        B = basis(new_points)  # B.shape == (dimension, len(new_points))
        return trG + np.sum(B**2, axis=0) * new_weights

    # def eta_l2(new_points: np.ndarray, new_weights: np.ndarray) -> np.ndarray:
    #     new_B_norms = B_norms[:, None] + basis(new_points) * new_weights
    #     new_B_norms /= len(samples) + 1
    #     assert new_B_norms.shape == (basis.dimension, len(new_points))
    #     return np.min(new_B_norms, axis=0)
    # TODO: This eta_l2 is probably not submodular. But maybe it has a submodularity ratio...
    # suboptimality ratio of min(f(x), b(x)), assuming both are modular
    # let f(xs) = trace(G(xs)) = sum_i trace(G([xs[i]]))
    # -->  f(xs + [x]) = f(xs) + f([x])
    # --> f(xs + [x]) - f(xs) = f([x]) .
    # Hence, f is modular.
    # min(f(xs + [x]), g(xs + [x])) = min(f(xs) + f([x]), g(xs) + g([x]))
    # # = 1/2 (f(xs) + f([x]) + g(xs) + g([x])) - 1/2 abs(f(xs) + f([x]) - (g(xs) + g([x])))
    # # >= 1/2 (f(xs) + g(xs)) - 1/2 abs(f(xs) - g(xs)) + 1/2 (f([x]) + g([x])) - 1/2 abs(f([x]) - g([x]))
    # # = min(f(xs), g(xs)) + min(f([x]), g([x]))
    # >= min(f(xs), g(xs)) + min(f([x]), g([x]))
    return eta_l2


def ensure_l2_stability(
    rng: np.random.Generator,
    core_basis: CoreBasis,
    discretisation: Float[np.ndarray, "discretisation"],
    target_suboptimality: float,
    samples: Float[np.ndarray, "sample_size dimension"],
    weights: Float[np.ndarray, " sample_size"],
    repetitions: int = 10,
) -> Float[np.ndarray, "sample_size dimension"]:
    dimension = np.reshape(core_basis.domain, (-1, 2)).shape[0]
    print("Ensuring greedy stability...")
    core_space_sampler = create_core_space_sampler(rng, core_basis, discretisation)
    candidate_size = int(4 * core_basis.dimension * np.log(4 * core_basis.dimension))
    # eta_l2 = fast_eta_l2_metric(core_basis, samples, weights)
    # # Note that eta_l2 = tr(G(sample + [new_point])) >= dimension * lambda_min(G(sample + [new_point])).
    # # So, we want eta_l2 to be at least dimension / target_suboptimality.
    # target_eta = core_basis.dimension / target_suboptimality
    # For the new eta_l2, we have
    suboptimality = lambda w, lambda_min: np.sqrt(1 + w**2 / lambda_min)
    # --> lambda_min / w^2 = 1 / (suboptimality^2 - 1)
    target_eta = 1 / (target_suboptimality**2 - 1)
    # eta = fast_eta_metric(product_kernel, core_basis_rkhs, np.reshape(samples, (-1, dimension)))
    # target_eta = 1 / target_suboptimality**2
    w = np.inf
    lambda_min = np.float64(0)
    test_vectors = np.eye(core_basis.dimension)
    B = core_basis(samples)
    while len(samples) < core_basis.dimension or suboptimality(w, lambda_min) > target_suboptimality:
        assert test_vectors.shape[0] == core_basis.dimension
        B_norms = (test_vectors.T @ B) ** 2 @ weights
        assert B_norms.shape == (test_vectors.shape[1],)
        assert np.all(B_norms >= 0)
        assert np.all(weights >= 0)
        max_w = np.max(weights, initial=0)

        def eta_l2(new_points: np.ndarray, new_weights: np.ndarray) -> np.ndarray:
            new_B_norms = B_norms[:, None] + (test_vectors.T @ core_basis(new_points)) ** 2 * new_weights
            new_B_norms /= len(samples) + 1
            assert new_B_norms.shape == (test_vectors.shape[1], len(new_points))
            assert np.all(new_B_norms >= 0)
            assert np.all(new_weights >= 0)
            new_max_w = np.maximum(max_w, new_weights)
            assert new_max_w.shape == (len(new_points),)
            return (
                np.min(new_B_norms, axis=0) / new_max_w**2
            )  # Note that we are estimating the factor for the square! --> new_max_w**2 instaed of new_max_w
            # # This is probably not submodular! So let's try the following relaxation:
            # new_B_norms = B_norms[:, None] / max(max_w, 1e-8) + (test_vectors.T @ core_basis(new_points))**2
            # new_B_norms /= len(samples) + 1
            # assert new_B_norms.shape == (test_vectors.shape[1], len(new_points))
            # assert np.all(new_B_norms >= 0), new_B_norms[~(new_B_norms >= 0)]
            # return np.min(new_B_norms, axis=0)
            # # NOTE: Since w <= 2, we can just leave this part out and enforce eta_l2 >= 2 * target_eta!
            # new_B_norms = B_norms[:, None] + (test_vectors.T @ core_basis(new_points))**2 * new_weights
            # new_B_norms /= len(samples) + 1
            # assert new_B_norms.shape == (test_vectors.shape[1], len(new_points))
            # assert np.all(new_B_norms >= 0)
            # return np.min(new_B_norms, axis=0) / 2
            # # NOTE: But on the other hand: Why should we?
            # The other bound is tighter and can not be made too much less submodular due to the bound weights <= 2.

        candidates, candidate_weights = draw_weighted_sequence(core_space_sampler, candidate_size)
        candidates, candidate_weights = np.asarray(candidates), np.asarray(candidate_weights)
        assert space in ["h1", "h10"]
        unif_candidates = rng.uniform(*l2_basis.domain, size=(candidate_size, dimension))
        christoffel = lambda points: core_basis_l2.core_space.christoffel(points, core_basis_l2.bases)
        unif_candidate_weights = 1 / christoffel(unif_candidates)
        candidates = np.concatenate([candidates, unif_candidates], axis=0)
        candidate_weights = 2 / (1 / np.concatenate([candidate_weights, unif_candidate_weights]) + 1)
        assert np.allclose(candidate_weights, 2 / (christoffel(candidates) + 1))
        assert np.all(candidate_weights <= 2) and np.all(weights <= 2)
        etas = eta_l2(candidates, candidate_weights)
        # etas = [eta(candidate) for candidate in candidates]
        order = np.argsort(-etas)
        # print(etas[order])
        sample_size = np.count_nonzero(np.cumsum(etas[order]) < target_eta) + 1
        sample_size = max(sample_size, core_basis.dimension - len(samples))
        samples = np.concatenate([samples, candidates[order[:sample_size]]], axis=0)
        weights = np.concatenate([weights, candidate_weights[order[:sample_size]]])
        assert samples.shape[0] == weights.shape[0]
        B = core_basis(samples)
        G = B * weights @ B.T / len(samples)
        es, vs = np.linalg.eigh(G)
        assert np.allclose(vs.T @ G @ vs, np.diag(es))
        lambda_min, lambda_max = max(np.min(es), np.float64(0)), np.max(es)
        print(f"  Sample size: {len(samples)}  |  L2-Gramian spectrum: [{lambda_min:.2e}, {lambda_max:.2e}]")
        w = np.max(weights)
        print(f"  Suboptimality: {suboptimality(w, lambda_min):.2e} < {target_suboptimality:.2e}")
        test_vectors = np.concatenate([test_vectors, vs[:, [np.argmin(es)]]], axis=1)
    assert np.isfinite(suboptimality(w, lambda_min)) and suboptimality(w, lambda_min) <= target_suboptimality

    print("TOOD: I could put a final loop here that subsamples from the newly selected points.")
    # What we could do, is have a first loop, where we don't update the sample but only the test_vectors.
    # This can be seen as a stochastic majorisation minimisation algorithm.
    # Then, we keep the test_vectors fixed and update the sample.
    # This can be seen as a (approximately) submodular maximisation algorithm.
    # OR: Instead of adding the chosen candidates to the sample, you store them in a separate list.
    #     Then, in the next iteration, you add this separate list to the new candidates.

    return samples, weights

    # selected = np.full(len(full_sample), False, dtype=bool)
    #     indices = np.arange(len(full_sample))
    #     value = -np.inf
    #     for selection_size in range(np.count_nonzero(selected), len(full_sample)):
    #         if value >= target:
    #             break
    #         selection_metric = selection_metric_factory(full_sample[selected])
    #         selected[indices[~selected][fast_greedy_step(selection_metric, full_sample[~selected])]] = True
    #         value = target_metric(full_sample[selected])
    #         print(f"  Sample size: {selection_size}  |  Target metric: {value:.2e} < {target:.2e}")
    #     return selected

    # # Actually, to have a bounded variance for the projector,
    # # we could simply multipy it with an independent Bernoulli with success probability
    # #     target_suboptimality / actual_suboptimality .


# def ensure_greedy_stability(
#     rng: np.random.Generator,
#     kernel: Kernel,
#     # core_basis_rkhs: CoreBasis,
#     # core_basis_l2: CoreBasis,
#     # discretisation: Float[np.ndarray, "discretisation"],
#     # target_suboptimality: float,
#     samples: Float[np.ndarray, "sample_size dimension"],
#     repetitions: int = 10,
# ) -> Float[np.ndarray, "sample_size dimension"]:
#     raise NotImplementedError()
#     dimension = np.reshape(core_basis_rkhs.domain, (-1, 2)).shape[0]
#     print("Ensuring greedy stability...")
#     eta_factory = lambda points: fast_eta_metric(kernel, core_basis_rkhs, np.reshape(points, (-1, dimension)))
#     lambda_ = lambda_metric(kernel, core_basis_rkhs)
#     # suboptimality = sqrt(1 + mu^2 tau^2) <= sqrt(1 + 1 / lambda), since tau <= 1
#     # --> lambda = 1 / (suboptimality^2 - 1)
#     lambda_target = 1 / (target_suboptimality**2 - 1)
#     core_space_sampler = create_core_space_sampler(rng, core_basis_l2, discretisation)
#     # candidates_size = int(repetitions * 4 * core_basis_l2.dimension * np.log(4 * core_basis_l2.dimension))
#     candidates_size = 2 * core_basis_l2.dimension
#     lambda_value = -np.inf
#     samples = list(samples)
#     while lambda_value < lambda_target:
#         print(f"  Initial candidates size: {candidates_size}")
#         candidates, candidate_weights = draw_weighted_sequence(core_space_sampler, candidates_size)
#         B = core_basis_l2(candidates)
#         G = B * candidate_weights @ B.T / candidates_size
#         es = np.linalg.eigvalsh(G)
#         print(f"  L2-Gramian spectrum: [{np.min(es):.2e}, {np.max(es):.2e}]")
#         # mu_l2 = np.sum((np.sum(abs(B * candidate_weights), axis=1) / candidates_size)**2)
#         mu_l2 = np.sum(np.sum(abs(B * candidate_weights), axis=1) ** 2) / candidates_size**2
#         print(f"  mu_l2: {mu_l2:.2e}")
#         print(core_basis_l2.dimension, core_basis_l2.dimension**2)
#         # K = kernel_matrix(kernel, candidates)
#         # es = np.linalg.eigvalsh(K)
#         # print(f"  Kernel matrix spectrum: [{np.min(es):.2e}, {np.max(es):.2e}]")
#         # B = core_basis_rkhs(candidates)
#         # es = np.linalg.eigvalsh(B @ np.linalg.inv(K) @ B.T)
#         # print(f"  Projected Gramian matrix spectrum: [{np.min(es):.2e}, {np.max(es):.2e}]")
#         # samples.append(candidates[fast_greedy_step(eta_factory(samples), candidates)])
#         # lambda_value = lambda_(samples)
#         # print(f"  Sample size: {len(samples)}  |  Suboptimality: {np.sqrt(1 + 1 / np.float64(lambda_value)):.2e} < {target_suboptimality:.2e}")
#         candidates_size *= 2
#         print("-" * 80)
#     exit()
#     samples = np.asarray(samples)
#     assert samples.ndim == 2 and samples.shape[1] == dimension
#     suboptimality = suboptimality_metric(product_kernel, core_basis_rkhs)
#     print(suboptimality(samples))
#     return samples


class CoreBasis(Basis):
    def __init__(self, tt: TensorTrain, bases: list[Basis]):
        assert tt.order == len(bases)
        self.core_space = TensorTrainCoreSpace(tt)
        self.bases = bases

    @property
    def dimension(self) -> int:
        return self.core_space.tensor_train.core.size

    @property
    def domain(self) -> list[tuple[float, float]]:
        return [b.domain for b in self.bases]

    def __call__(
        self, points: Float[np.ndarray, "sample_size dimension"]
    ) -> Float[np.ndarray, "core_dimension sample_size"]:
        points = np.asarray(points)
        assert points.ndim == 2 and points.shape[1] == self.core_space.tensor_train.order
        bs = [b(points[:, m]) for m, b in enumerate(self.bases)]
        ln, en, rn = self.core_space.evaluate(bs)
        return contract("ln, en, rn -> lern", ln, en, rn).reshape(self.dimension, points.shape[0])


def evaluate(
    tt: TensorTrain, bases: list[Basis], points: Float[np.ndarray, "sample_size dimension"]
) -> Float[np.ndarray, " sample_size"]:
    return CoreBasis(tt, bases)(points).T @ tt.core.reshape(-1)


def create_core_space_sampler(
    rng: np.random.Generator, core_basis: CoreBasis, discretisation: Float[np.ndarray, "discretisation"]
) -> Sampler:
    core_space = core_basis.core_space
    order = core_space.tensor_train.order

    def core_space_sampler(
        conditioned_on: Optional[tuple[list[Float[np.ndarray, "dimension"]], list[float]]] = None
    ) -> tuple[Float[np.ndarray, "dimension"], float]:
        sample = core_space.christoffel_sample(
            rng, core_basis.bases, [rho] * order, [discretisation] * order, sample_size=1
        )
        sample = np.array(sample, dtype=float)
        weight = 1 / core_space.christoffel(sample, core_basis.bases)
        return sample[0], weight[0]

    return core_space_sampler


def least_squares(points: np.ndarray, weights: np.ndarray, values: np.ndarray, basis: Basis) -> np.ndarray:
    assert points.ndim == 2 and values.ndim == 1 and weights.ndim == 1
    assert len(points) == len(values) == len(weights)
    # B * weights @ B.T @ result = (B * weights @ values)
    B = basis(points) * np.sqrt(weights)
    v = values * np.sqrt(weights)
    return np.linalg.lstsq(B.T, v, rcond=None)[0]


def quasi_projection(points: np.ndarray, values: np.ndarray, weights: np.ndarray, basis: Basis):
    assert points.ndim >= 1 and values.ndim == 1 and weights.ndim == 1
    assert len(points) == len(values) == len(weights)
    dimension = np.reshape(basis.domain, (-1, 2)).shape[0]
    if points.ndim == 1 and dimension != 1:
        raise ValueError(f"dimension mismatch: input dimension is 1 but space dimension is {dimension}")
    elif points.shape[1] != dimension:
        raise ValueError(f"dimension mismatch: input dimension is {points.shape[1]} but space dimension is {dimension}")
    return basis(points) @ (values * weights) / len(points)


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    from rkhs_1d import H10Kernel, H1Kernel, H1GaussKernel  # , kernel_matrix
    from rkhs_nd import TensorProductKernel
    from basis_1d import (
        compute_discrete_gramian,
        enforce_zero_trace,
        orthonormalise,
        orthogonalise,
        MonomialBasis,
        FourierBasis,
        SinBasis,
    )
    # from greedy_subsampling import fast_eta_metric, lambda_metric, suboptimality_metric

    # from least_squares import optimal_least_squares

    rng = np.random.default_rng(0)
    suffix = set()

    # dimension = 5
    dimension = 10
    basis_name = "polynomial"
    # basis_name = "fourier"
    space = ["h10", "h1", "h1gauss"][1]

    # target_suboptimality = 50000
    # target_suboptimality = 50
    target_suboptimality = 5
    suffix |= {f"ts{target_suboptimality}"}

    debiasing_sample_size = 1
    # max_iteration = 2000
    max_iteration = 500
    # max_iteration = 100

    all_parameters = {
        "draw_for_stability",
        "draw_for_stability_bound",
        "update_in_stable_space",
        "use_stable_projection",
        "use_debiasing",
    }

    # Original algorithm
    # used_parameters = {"draw_for_stability", "use_debiasing"}
    used_parameters = {"draw_for_stability_bound", "use_debiasing"}

    # # Try to use fewer samples by updating only in the subspace where G(x) is stable.
    # #     Note that this is a generalisation of the conditionally stable projector from Cohen and Migliorati.
    # #     They use the empirical projector only if it is stable.
    # #     We use the stable subspace, which may be zero-dimensional.
    # #     One problem is, that we don't have any guarantees about |(I-P) grad| <= |P grad|.
    # #     This means that even though the update is stable, it may be detrimental.
    # used_parameters = {"update_in_stable_space", "use_debiasing"}

    # # Try to use fewer samples by updating only when the entire G(x) is stable.
    # used_parameters = {"use_stable_projection", "use_debiasing"}

    # Reference algorithm (only use the RKHS projection)
    # used_parameters = {"draw_for_stability",}
    # used_parameters = {"draw_for_stability_bound",}

    assert used_parameters <= all_parameters
    for parameter in used_parameters:
        globals()[parameter] = True
    for parameter in all_parameters - used_parameters:
        globals()[parameter] = False

    draw_for_stability: bool
    draw_for_stability_bound: bool
    use_debiasing: bool
    update_in_stable_space: bool
    use_stable_projection: bool
    assert draw_for_stability or draw_for_stability_bound or use_debiasing

    order = 2
    # order = 3
    # order = 4
    # order = 5
    ranks = [2, 4, 6]

    # def target(points: np.ndarray) -> np.ndarray:
    #     # Corner peak in two dimensions
    #     cs = np.array([3.0, 5.0])
    #     points = np.asarray(points)
    #     assert points.ndim == 2 and points.shape[1] == len(cs)
    #     assert np.allclose(np.asarray(rkhs_basis.domain), [-1, 1])
    #     points = (points + 1) / 2  # transform points to interval [0, 1]
    #     return (1 + points @ cs)**(-(len(cs) + 1))
    # target.__name__ = "corner_peak"

    # def target(points: np.ndarray) -> np.ndarray:
    #     points = np.asarray(points)
    #     assert points.ndim == 2 and points.shape[1] == 2
    #     return np.cos(points[:, 0] + points[:, 1]) * np.exp(points[:, 0] * points[:, 1])
    # target.__name__ = "cheng_sandu"

    def target(points: np.ndarray) -> np.ndarray:
        # Anthony's test function
        points = np.asarray(points)
        assert np.allclose(np.asarray(rkhs_basis.domain), [-1, 1])
        points = (points + 1) / 2  # transform points to interval [0, 1]
        return 1 / (1 + np.sum(points, axis=1))

    target.__name__ = "anthony"

    # def target(points: np.ndarray) -> np.ndarray:
    #     points = np.asarray(points)
    #     # cs = np.ones(l2_basis.dimension)
    #     # cs[max(ranks):] = 0
    #     cs = 1 / np.arange(1, l2_basis.dimension + 1)
    #     bs = 1
    #     for ps in points.T:
    #         bs *= l2_basis(ps)
    #     assert bs.shape == (l2_basis.dimension, len(points))
    #     return cs @ bs
    # target.__name__ = "full_rank"

    print(f"Approximating: {target.__name__}")
    print("Algorithm parameters:")
    max_parameter_len = max(len(p) for p in all_parameters)
    for parameter in sorted(all_parameters):
        print(f"    {parameter:<{max_parameter_len}s} = {globals()[parameter]}")

    if space == "h10":

        def rho(x):
            return np.full_like(x, fill_value=0.5, dtype=float)

        rkhs_kernel = H10Kernel((-1, 1))
        if basis_name == "polynomial":
            initial_basis = MonomialBasis(dimension, domain=(-1, 1))
        elif basis_name == "fourier":
            initial_basis = SinBasis(dimension, domain=(-1, 1))
        else:
            raise NotImplementedError()
        initial_basis = enforce_zero_trace(initial_basis)
    elif space == "h1":

        def rho(x):
            return np.full_like(x, fill_value=0.5, dtype=float)

        rkhs_kernel = H1Kernel((-1, 1))
        if basis_name == "polynomial":
            initial_basis = MonomialBasis(dimension, domain=(-1, 1))
        elif basis_name == "fourier":
            initial_basis = FourierBasis(dimension, domain=(-1, 1))
        else:
            raise NotImplementedError()
    elif space == "h1gauss":
        assert dimension <= 10

        def rho(x):
            return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

        rkhs_kernel = H1GaussKernel((-8, 8))
        if basis_name == "polynomial":
            initial_basis = MonomialBasis(dimension, domain=(-8, 8))
        elif basis_name == "fourier":
            initial_basis = FourierBasis(dimension, domain=(-8, 8))
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    product_kernel = TensorProductKernel(*(rkhs_kernel,) * order)

    # discretisation = np.linspace(*initial_basis.domain, 10000)
    # discretisation = np.linspace(*initial_basis.domain, 1000)
    discretisation = np.linspace(*initial_basis.domain, 100)

    discrete_rkhs_gramian = compute_discrete_gramian(space, initial_basis.domain, 2**13)
    rkhs_basis = orthonormalise(initial_basis, *discrete_rkhs_gramian)
    if space == "h1gauss":
        discrete_l2_gramian = compute_discrete_gramian("l2gauss", rkhs_basis.domain, 2**13)
    else:
        discrete_l2_gramian = compute_discrete_gramian("l2", rkhs_basis.domain, 2**13)
    rkhs_basis, l2_norms = orthogonalise(rkhs_basis, *discrete_l2_gramian)
    l2_basis = TransformedBasis(np.diag(1 / l2_norms), rkhs_basis)
    rkhs_to_l2 = np.diag(l2_norms)

    # M, xs = discrete_l2_gramian
    # G = l2_basis(xs)
    # G = G @ M @ G.T
    # print(f"L2-Gramian distance from idenity: {np.linalg.norm(G - np.eye(G.shape[0]), ord=2):.2e}")
    # exit()

    if space in ["h1", "h10"]:
        test_sample = rng.uniform(-1, 1, size=(10000, order))
    elif space == "h1gauss":
        test_sample = rng.standard_normal(size=(10000, order))
    else:
        raise NotImplementedError()
    test_values = target(test_sample)

    def compute_test_error(tt: TensorTrain) -> float:
        # prediction = evaluate(tt, [rkhs_basis]*tt.order, test_sample)
        prediction = evaluate(tt, [l2_basis] * tt.order, test_sample)
        assert prediction.shape == test_values.shape
        return np.linalg.norm(prediction - test_values, ord=2) / np.linalg.norm(test_values, ord=2)

    plt.style.use("classic")
    fig, ax = plt.subplots(1, 1)
    # fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    for rank in ranks:
        print(f"Initialise TT of rank {rank}")
        tt = TensorTrain.random(rng, [dimension] * order, [1] + [rank] * (order - 1) + [1])

        full_sample = np.empty(shape=(0, order), dtype=float)
        full_weights = np.empty(shape=(0,), dtype=float)
        full_values = np.empty(shape=(0,), dtype=float)
        core_position = -1
        errors = []
        sample_sizes = []
        assert tt.core_position == 0
        all_directions = {0: "right", tt.order - 1: "left"}
        sweeps = 0
        test_error = compute_test_error(tt)
        for it in range(1, max_iteration + 1):
            sweeps += tt.core_position == 0
            # step_size = 1 / np.sqrt(it)
            # step_size = 1; postfix |= {"ss1",}
            step_size = 1 / np.sqrt(sweeps)
            suffix |= {"sssw"}
            print(
                f"Iteration: {it}"
                f"  |  Sweep: {sweeps}"
                f"  |  Core position: {tt.core_position}"
                f"  |  Step size: {step_size:.2e}"
            )

            # core_basis_rkhs = CoreBasis(tt, [rkhs_basis]*tt.order)
            # eta_factory = lambda points: fast_eta_metric(product_kernel, core_basis_rkhs, points)
            # lambda_ = lambda_metric(product_kernel, core_basis_rkhs)
            # suboptimality = suboptimality_metric(product_kernel, core_basis_rkhs)

            # core_basis_l2 = CoreBasis(tt.transform([rkhs_to_l2] * tt.order), [l2_basis]*tt.order)
            core_basis_l2 = CoreBasis(tt, [l2_basis] * tt.order)

            def l2_suboptimality(sample, weights):
                if len(sample) < core_basis_l2.dimension:
                    return np.inf
                B = core_basis_l2(sample)
                assert B.shape == (core_basis_l2.dimension, len(sample))
                G = B * weights @ B.T / len(sample)
                assert np.all(weights >= 0)
                return np.sqrt(1 + np.max(weights) ** 2 / np.linalg.norm(G, ord=-2))

            # current_suboptimality = suboptimality(full_sample)
            current_suboptimality = l2_suboptimality(full_sample, full_weights)
            print(
                f"Sample size: {len(full_sample)}"
                f"  |  Oversampling: {len(full_sample) / tt.parameters:.2f}"
                f"  |  Suboptimality factor: {current_suboptimality:.1f} < {target_suboptimality:.1f}"
            )
            if draw_for_stability or draw_for_stability_bound:
                if current_suboptimality > target_suboptimality:
                    # suboptimality = sqrt(1 + mu^2 tau^2) --> mu^2 = (suboptimality^2 - 1) / tau^2 --> lambda = 1 / mu^2
                    # candidates = ensure_stability(core_basis_rkhs, lambda_, 1 / (target_suboptimality**2 - 1))
                    # candidates = np.concatenate([full_sample, candidates], axis=0)
                    # selected = np.full(len(candidates), False, dtype=bool)
                    # selected[:len(full_sample)] = True
                    # selection_size = max(len(full_sample) + 2, 2 * core_basis_rkhs.dimension)
                    # if draw_for_stability:
                    #     selected = greedy_draw(eta_factory, selection_size, candidates, selected)
                    # else:
                    #     assert draw_for_stability_bound
                    #     selected = greedy_bound(eta_factory, lambda_, 1 / (target_suboptimality**2 - 1), candidates, selected)
                    # assert np.all(selected[:len(full_sample)])
                    # new_selected = selected[len(full_sample):]
                    # full_values = np.concatenate([full_values, target(candidates[len(full_sample):][new_selected])], axis=0)
                    # full_sample = candidates[selected]
                    assert draw_for_stability_bound
                    # core_basis_l2 = CoreBasis(tt.transform([rkhs_to_l2] * tt.order), [l2_basis]*tt.order)
                    # # full_sample = ensure_greedy_stability(rng, product_kernel, core_basis_l2, discretisation, target_suboptimality, full_sample)
                    # full_sample = ensure_greedy_stability(rng, product_kernel, full_sample)
                    # # full_sample = ensure_greedy_stability(rng, product_kernel, core_basis_rkhs, discretisation, target_suboptimality, full_sample)
                    # full_values = np.concatenate([full_values, target(full_sample[len(full_values):])], axis=0)
                    # current_suboptimality = suboptimality(full_sample)
                    full_sample, full_weights = ensure_l2_stability(rng, core_basis_l2, discretisation, target_suboptimality, full_sample, full_weights)
                    full_values = np.concatenate([full_values, target(full_sample[len(full_values):])], axis=0)
                    assert full_values.ndim == 1 and full_sample.shape == (len(full_values), order)
                    current_suboptimality = l2_suboptimality(full_sample, full_weights)
                    print(f"Sample size: {len(full_sample)}  |  Oversampling: {len(full_sample) / tt.parameters:.2f}  |  Suboptimality factor: {current_suboptimality:.1f} < {target_suboptimality:.1f}")

            # K = kernel_matrix(product_kernel, full_sample)
            # es = np.linalg.eigvalsh(K)
            # while np.any(es < 1e-8):
            #     print("WARNING: kernel matrix is not positive definite")
            #     print("Removing duplicates...")
            #     ds = np.linalg.norm(full_sample[:, None] - full_sample[None, :], axis=2)
            #     assert np.all(ds >= 0) and np.all(ds == ds.T)
            #     js, ks = np.where(ds < 1e-8)
            #     js = np.max(js[js != ks])
            #     print(f"  Removing sample {js+1} / {len(full_sample)}")
            #     full_sample = np.delete(full_sample, js, axis=0)
            #     full_values = np.delete(full_values, js, axis=0)
            #     K = kernel_matrix(product_kernel, full_sample)
            #     es = np.linalg.eigvalsh(K)

            # if update_in_stable_space:
            #     print("Computing stable directions...")
            #     stability_threshold = 10
            #     print(f"Stability threshold: {stability_threshold:.1f}")
            #     K = product_kernel(full_sample[:, None], full_sample[None, :])
            #     assert K.shape == (len(full_sample), len(full_sample))
            #     M = core_basis_rkhs(full_sample)
            #     assert M.shape == (core_basis_rkhs.dimension, len(full_sample))
            #     # We want to compute M @ inv(K) @ M.T
            #     G = np.linalg.lstsq(K, M.T, rcond=None)[0]
            #     assert G.shape == (len(full_sample), core_basis_rkhs.dimension)
            #     G = M @ G
            #     es, vs = np.linalg.eigh(G)
            #     assert np.allclose(vs * es @ vs.T, G)
            #     es = np.maximum(es, 0)
            #     # The stability condition is (1 + mu^2 tau^2)^0.5 <= stability_threshold with
            #     #     mu^2 = 1 / lambda_min(G) and
            #     #     tau^2 = min(4 ||G - I||_F^2, 1) .
            #     # It holds that
            #     #     mu^2 = 1 / min(es)
            #     #     tau^2 = min(4 sum((es - 1)^2), 1) .
            #     stab = lambda es: np.sqrt(1 + min(4 * np.sum((es - 1)**2), 1) / np.min(es))
            #     indices = np.arange(len(es))
            #     mask = np.full(len(es), True)
            #     while np.any(mask) and stab(es[mask]) > stability_threshold:
            #         rm_idx = np.argmin(es[mask])
            #         mask[indices[mask][rm_idx]] = False
            #     # P = vs[:, mask] @ vs[:, mask].T
            #     stable_space_dimension = np.count_nonzero(mask)
            #     P = vs[:, mask].T
            #     stable_basis = TransformedBasis(P, core_basis_rkhs)
            #     print(f"Stable space dimension: {stable_space_dimension} / {core_basis_rkhs.dimension}")

            def gradient(points: np.ndarray, values: np.ndarray) -> np.ndarray:
                # L(v) = 0.5 |v - u|^2  -->  grad(L(v)) = v - u
                # v - grad(L(v)) = v - (v - u) = u ✅
                assert values.ndim == 1 and points.shape == values.shape + (order,)
                # prediction = evaluate(tt, [rkhs_basis]*tt.order, points)
                prediction = evaluate(tt, [l2_basis] * tt.order, points)
                assert prediction.shape == values.shape
                return prediction - values

            full_grad = gradient(full_sample, full_values)
            assert np.all(np.isfinite(full_grad))
            assert not (update_in_stable_space or use_stable_projection)

            assert np.allclose(
                evaluate(tt, [l2_basis] * tt.order, test_sample), tt.core.reshape(-1) @ core_basis_l2(test_sample)
            )

            print(f"Relative test set error: {test_error:.2e}")
            print("Perform LS update...")
            print(f"  Least squares variance: {np.sqrt(1 + current_suboptimality) * core_basis_l2.dimension:.2e}")
            update_core = least_squares(full_sample, full_weights, full_grad, core_basis_l2)
            tt.core -= step_size * update_core.reshape(tt.core.shape)
            test_error = compute_test_error(tt)
            print(f"  Relative test set error: {test_error:.2e}")

            # ls_probability = min(target_suboptimality / l2_suboptimality(full_sample, full_weights), 1)
            # print(f"  Least squares probability: {ls_probability:.2f}")
            # update_core = rng.binomial(1, ls_probability) * least_squares(full_sample, full_weights, full_grad, core_basis_l2)
            # ls_scaling = min(target_suboptimality / l2_suboptimality(full_sample, full_weights), 1)

            # ls_scaling = 1 / np.sqrt(1 + l2_suboptimality(full_sample, full_weights))
            # print(f"  Least squares scaling: {ls_scaling:.2f}")
            # update_core = ls_scaling * least_squares(full_sample, full_weights, full_grad, core_basis_l2)


            if use_debiasing:
                print("Perform debiasing update...")
                debiasing_sample, debiasing_weights = draw_weighted_sequence(create_core_space_sampler(rng, core_basis_l2, discretisation), debiasing_sample_size)
                debiasing_sample, debiasing_weights = np.asarray(debiasing_sample), np.asarray(debiasing_weights)
                assert space in ["h1", "h10"]
                print(f"  Debiasing variance: {core_basis_l2.dimension:.2e}")
                mask = rng.binomial(1, 0.5, size=len(debiasing_sample)).astype(bool)
                debiasing_sample[mask] = rng.uniform(*l2_basis.domain, size=(np.count_nonzero(mask), order))
                christoffel = lambda points: core_basis_l2.core_space.christoffel(points, core_basis_l2.bases)
                debiasing_weights = 2 / (christoffel(debiasing_sample) + 1)
                assert debiasing_weights.shape == (len(debiasing_sample),)
                debiasing_values = target(debiasing_sample)
                debiasing_grad = gradient(debiasing_sample, debiasing_values)
                assert np.all(np.isfinite(debiasing_grad))
                projected_debiasing_grad = update_core @ core_basis_l2(debiasing_sample)
                debiasing_core = quasi_projection(debiasing_sample, debiasing_grad - projected_debiasing_grad, debiasing_weights, core_basis_l2)
                assert np.all(np.isfinite(debiasing_core))
                tt.core -= step_size * update_core.reshape(tt.core.shape)
                test_error = compute_test_error(tt)
                print(f"  Relative test set error: {test_error:.2e}")

            # full_grad = gradient(full_sample, full_values)
            # assert np.all(np.isfinite(full_grad))
            # if update_in_stable_space:
            #     update_core = P.T @ optimal_least_squares(full_sample, full_grad, product_kernel, stable_basis)
            # elif use_stable_projection:
            #     if current_suboptimality <= target_suboptimality:
            #         update_core = optimal_least_squares(full_sample, full_grad, product_kernel, core_basis_rkhs)
            #     else:
            #         update_core = np.full(tt[core_position].size, fill_value=0.0, dtype=float)
            # else:
            #     update_core = optimal_least_squares(full_sample, full_grad, product_kernel, core_basis_rkhs)
            # assert np.all(np.isfinite(update_core))

            # if use_debiasing:
            #     print("Draw debiasing sample...")
            #     core_space_l2 = TensorTrainCoreSpace(tt)
            #     core_space_l2, (tl, te, tr) = core_space_l2.transform([rkhs_to_l2] * tt.order)
            #     core_basis_l2 = CoreBasis(core_space_l2.tensor_train, [l2_basis]*tt.order)
            #     debiasing_sample, debiasing_weights = draw_weighted_sequence(create_core_space_sampler(rng, core_basis_l2, discretisation), debiasing_sample_size)
            #     debiasing_sample, debiasing_weights = np.asarray(debiasing_sample), np.asarray(debiasing_weights)
            #     assert debiasing_weights.shape == (len(debiasing_sample),)
            #     debiasing_values = target(debiasing_sample)
            #     debiasing_grad = gradient(debiasing_sample, debiasing_values)
            #     assert np.all(np.isfinite(debiasing_grad))
            #     residual = debiasing_grad - core_basis_rkhs(debiasing_sample).T @ update_core
            #     assert np.all(np.isfinite(residual))

            #     # debiasing_core = quasi_projection(debiasing_sample, residual, debiasing_weights, core_basis_rkhs)

            #     # NOTE: Ideally, we would like to compute the quasi-projection w.r.t. the L2 basis and then perform a change of basis to the RKHS basis.
            #     # Consider a function v(x) = c @ b(x) where b is a V-orthonormal basis.
            #     # Moreover, suppose that B = M @ b is a change of basis to an L2-orthonormal basis.
            #     # The quasi-projection of a function u in L2 with respect to the basis B is given by g = u(x) @ B(x).T @ B.
            #     # To add the function g to v, we have to perform a change of basis first:
            #     #     g = (u(x) @ B(x).T) @ B = (u(x) @ b(x).T) @ M.T @ M @ b.
            #     # Therefore
            #     #     f + g = c + (u(x) @ b(x).T) @ M.T @ M .
            #     # This means that we can perform a quasi-projection w.r.t. the basis B by performing a quasi-projection w.r.t.
            #     # the basis b and multiplying the resulting coefficients with M.T @ M.
            #     # We know that the TT satisfies M.T @ M comes from the core space transform.
            #     debiasing_core = quasi_projection(debiasing_sample, residual, debiasing_weights, core_basis_l2)
            #     debiasing_core = contract("ler, Ll, Ee, Rr -> LER", debiasing_core.reshape(tt.core.shape), tl.T @ tl, te.T @ te, tr.T @ tr).reshape(-1)

            #     assert np.all(np.isfinite(debiasing_core))
            #     update_core += debiasing_core

            # tt.core -= step_size * update_core.reshape(tt.core.shape)

            errors.append(test_error)
            sample_sizes.append(len(full_sample))

            svs = tt.singular_values()
            erank_ps = lambda s: s / np.sum(s)
            erank = lambda s: np.exp(-np.sum(erank_ps(s) * np.log(erank_ps(s))))
            erks = "[" + ", ".join(f"{erank(svs):.2f}" for svs in svs) + "]"
            print(f"Effective ranks: {erks}")

            max_derivative = np.log(np.max(errors[-50:])) - np.log(np.min(errors[-50:]))
            print(f"Max derivative: {max_derivative:.2e} > {0.05:.2e}")
            print()
            if len(errors) >= 50 and max_derivative <= 0.05:
                break

            if tt.core_position in all_directions:
                direction = all_directions[tt.core_position]
            core_move_test_values = evaluate(tt, [l2_basis] * tt.order, test_sample)
            tt.move_core(direction)
            assert np.allclose(core_move_test_values, evaluate(tt, [l2_basis] * tt.order, test_sample))

            if use_debiasing:
                full_sample = np.concatenate([full_sample, debiasing_sample], axis=0)
                full_weights = np.concatenate([full_weights, debiasing_weights], axis=0)
                full_values = np.concatenate([full_values, debiasing_values], axis=0)

        test_error = compute_test_error(tt)
        errors.append(test_error)
        sample_sizes.append(len(full_sample))
        print(f"Relative test set error: {test_error:.2e}")

        if rank == 2:
            color = "b"
        elif rank == 4:
            color = "r"
        elif rank == 6:
            # color = "g"
            color = (0, 1, 0)
        else:
            color = "k"
        ax.plot(
            np.asarray(sample_sizes) / tt.parameters,
            errors,
            ">--",
            color=color,
            fillstyle="none",
            linewidth=1.5,
            markeredgewidth=1.5,
            markersize=6,
            label=f"$r = {rank}$",
        )
        print()

    ax.set_yscale("log")
    ax.set_xlabel("nb. evaluations / nb. parameters")
    ax.set_ylabel("error")
    ax.legend()

    plot_directory = Path(__file__).parent / "plot"
    plot_directory.mkdir(exist_ok=True)
    plot_file = f"als_{order}_{target.__name__}"
    plot_file += "_" + "_".join(sorted(used_parameters))
    plot_file += ("_" if suffix else "") + "_".join(sorted(suffix))
    plot_path = plot_directory / f"{plot_file}.png"
    print("Saving sample statistics plot to", plot_path)
    plt.savefig(plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True)
    # plt.show()
    plt.close(fig)
