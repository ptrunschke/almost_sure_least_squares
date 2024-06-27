from __future__ import annotations
from jaxtyping import Float
import numpy as np
from opt_einsum import contract

from basis_1d import TransformedBasis, Basis
from greedy_subsampling import FastMetric
from tensor_train import TensorTrain, TensorTrainCoreSpace


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
    discretisation: Float[np.ndarray, " discretisation"],
    target_suboptimality: float,
    samples: Float[np.ndarray, "sample_size dimension"],
    weights: Float[np.ndarray, " sample_size"],
    repetitions: int = 10,
) -> Float[np.ndarray, "sample_size dimension"]:
    dimension = np.reshape(core_basis.domain, (-1, 2)).shape[0]
    samples = np.reshape(samples, (-1, dimension))
    weights = np.asarray(weights)
    assert len(samples) == len(weights)
    print("Ensuring greedy stability...")
    candidate_size = int(repetitions * 4 * core_basis.dimension * np.log(4 * core_basis.dimension))
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

        christoffel_candidates = core_basis.core_space.christoffel_sample(
            rng,
            core_basis.bases,
            [rho] * core_basis.core_space.tensor_train.order,
            [discretisation] * core_basis.core_space.tensor_train.order,
            sample_size=candidate_size,
            stratified=True,
        )
        assert space in ["h1", "h10"]
        uniform_candidates = rng.uniform(*l2_basis.domain, size=(candidate_size, dimension))
        candidates = np.concatenate([christoffel_candidates, uniform_candidates], axis=0)
        christoffel = lambda points: core_basis_l2.core_space.christoffel(points, core_basis_l2.bases)
        candidate_weights = 2 / (christoffel(candidates) + 1)
        assert np.all(candidate_weights <= 2) and np.all(weights <= 2)
        assert candidate_weights.shape == (2 * candidate_size,)

        etas = eta_l2(candidates, candidate_weights)
        order = np.argsort(-etas)
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


def least_squares(points: np.ndarray, weights: np.ndarray, values: np.ndarray, basis: Basis) -> np.ndarray:
    points = np.asarray(points)
    weights = np.asarray(weights)
    values = np.asarray(values)
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

    rng = np.random.default_rng(0)
    suffix = set()

    dimension = 10
    basis_name = ["polynomial", "fourier"][0]
    space = ["h10", "h1", "h1gauss"][1]

    target_suboptimality = 5
    suffix |= {f"ts{target_suboptimality}"}

    # # Original algorithm
    # used_parameters = {"draw_for_stability_bound", "use_debiasing"}
    # # used_parameters = {"draw_for_stability_bound", "use_debiasing", "initialisation_sweep"}
    # # used_parameters = {"draw_for_stability_bound", "initialisation_sweep"}
    # # TODO: Diese Parameter machen klar, wo das Problem liegt.
    # # Es ist nicht verwunderlich, dass wir nicht unter einen Fehler von 1 kommen, wenn wir keine neuen Samplepunkte ziehen.
    # # Die Frage ist: Warum? Wir wissen, dass wir eigentlich quasi-optimal sein muessten.
    # # Aber halt nur bezuglich der Unendlich-Norm (bzw der n-norm). Die ist 


    order = 5
    # ranks = [2, 4, 6]
    ranks = [4]

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


    # scheme_str = "3xLSD-NONE_1000xLS-QP"
    # scheme_str = "1000xCLS-NONE"
    # scheme_str = "100xNONE-QP"
    # scheme_str = "1000xNONE-QPD"
    # scheme_str = "100xCRLS-NONE"
    scheme_str = "5xCRLS-NONE_100xCRLS-QPD"
    # CLS: conditioned least squares (sample size adapted to dimension)
    # CRLS: conditioned recycling least squares
    # LSSS: least squares in stable space
    # NONE: do nothing
    # QP: quasi-projection, sample size 1
    # QPD: quasi-projection, sample size == dimension

    def split_schemes(scheme_str: str) -> list[str]:
        return scheme_str.split("_")

    def parse_scheme(scheme_str: str) -> tuple[int, str, str]:
        count, ids = scheme_str.split("x")
        count = int(count)
        lsid, qpid = ids.split("-")
        assert lsid.isalpha() and lsid.upper() == lsid 
        assert qpid.isalpha() and qpid.upper() == qpid
        return count, lsid, qpid

    def iter_schemes(scheme_str: str):
        for scheme in split_schemes(scheme_str):
            count, lsid, qpid = parse_scheme(scheme)
            for _ in range (count):
                yield lsid, qpid

    print(f"Approximating: {target.__name__}")
    print("Optimisation scheme:")
    for scheme in split_schemes(scheme_str):
        count, lsid, qpid = parse_scheme(scheme)
        print(f"  {count} ✖️ ({lsid} + {qpid})")

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

    M, xs = discrete_l2_gramian
    G = l2_basis(xs)
    G = G @ M @ G.T
    assert np.linalg.norm(G - np.eye(G.shape[0]), ord=2) < 1e-8

    if space in ["h1", "h10"]:
        test_sample = rng.uniform(-1, 1, size=(10000, order))
    elif space == "h1gauss":
        test_sample = rng.standard_normal(size=(10000, order))
    else:
        raise NotImplementedError()
    test_values = target(test_sample)

    def compute_test_error(tt: TensorTrain) -> float:
        prediction = evaluate(tt, [l2_basis] * tt.order, test_sample)
        assert prediction.shape == test_values.shape
        return np.linalg.norm(prediction - test_values, ord=2) / np.linalg.norm(test_values, ord=2)

    plt.style.use("classic")
    fig, ax = plt.subplots(1, 1)
    for rank in ranks:
        print(f"Initialise TT of rank {rank}")
        tt = TensorTrain.random(rng, [dimension] * order, [1] + [rank] * (order - 1) + [1])

        full_sample = []
        full_weights = []
        full_values = []
        core_position = -1
        errors = []
        sample_sizes = []
        assert tt.core_position == 0
        all_directions = {0: "right", tt.order - 1: "left"}
        sweeps = 0
        test_error = compute_test_error(tt)
        for it, (lsid, qpid) in enumerate(iter_schemes(scheme_str), start=1):
            sweeps += tt.core_position == 0
            # step_size = 1 / np.sqrt(it)
            # suffix |= {"srt"}
            # step_size = 1
            # suffix |= {"s1"}
            step_size = 1 / np.sqrt(sweeps)
            suffix |= {"ssrt"}
            print(
                f"Iteration: {it}"
                f"  |  Sweep: {sweeps}"
                f"  |  Core position: {tt.core_position}"
                f"  |  Step size: {step_size:.2e}"
            )

            core_basis_l2 = CoreBasis(tt, [l2_basis] * tt.order)
            assert core_basis_l2.dimension == tt.core.size
            christoffel = lambda points: core_basis_l2.core_space.christoffel(points, core_basis_l2.bases)

            def gradient(points: np.ndarray, values: np.ndarray) -> np.ndarray:
                points = np.asarray(points)
                values = np.asarray(values)
                # L(v) = 0.5 |v - u|^2  -->  grad(L(v)) = v - u
                # v - grad(L(v)) = v - (v - u) = u ✅
                assert values.ndim == 1 and points.shape == values.shape + (order,)
                # prediction = evaluate(tt, [rkhs_basis]*tt.order, points)
                prediction = evaluate(tt, [l2_basis] * tt.order, points)
                assert prediction.shape == values.shape
                return prediction - values

            if lsid == "CLS":
                print("Performing conditioned least squares update...")

                least_squares_sample_size  = int(4 * core_basis_l2.dimension * np.log(4 * core_basis_l2.dimension))
                l_min = 0
                while l_min < 0.5:
                    least_squares_sample = core_basis_l2.core_space.christoffel_sample(
                        rng,
                        core_basis_l2.bases,
                        [rho] * order,
                        [discretisation] * order,
                        sample_size=least_squares_sample_size,
                        stratified=True,
                    )
                    assert space in ["h1", "h10"]
                    mask = rng.binomial(1, 0.5, size=len(least_squares_sample)).astype(bool)
                    least_squares_sample[mask] = rng.uniform(*l2_basis.domain, size=(np.count_nonzero(mask), order))
                    least_squares_weights = 2 / (christoffel(least_squares_sample) + 1)
                    assert np.all(least_squares_weights <= 2)

                    B = core_basis_l2(least_squares_sample)
                    assert B.shape == (core_basis_l2.dimension, len(least_squares_sample))
                    G = B * least_squares_weights @ B.T / len(least_squares_sample)
                    es = np.linalg.eigvalsh(G)
                    es = np.maximum(es, 0)
                    l_min = np.min(es)

                    # sample_factor = (1 + np.sqrt(np.max(es) / np.min(es)))**2 / np.min(es)
                    # norm_factor = least_squares_gradient**2 @ least_squares_weights / len(least_squares_sample)
                    # adv = sample_factor * norm_factor
                    # adv *= 2 * core_basis_l2.dimension / least_squares_sample_size
                    # # print(f"  Least squares variance: {np.sqrt(1 + current_suboptimality) * core_basis_l2.dimension:.2e}")
                    # print(f"  Additive debiasing variance: {adv:.2e} (sample factor: {sample_factor:.2e}, norm factor: {norm_factor:.2e})")
                assert least_squares_weights.shape == (len(least_squares_sample),)
                least_squares_values = target(least_squares_sample)
                least_squares_gradient = gradient(least_squares_sample, least_squares_values)
                assert np.all(np.isfinite(least_squares_gradient))

                g_norm = np.linalg.norm(least_squares_gradient, ord=2)**2
                print(f"  Additive debiasing variance: {step_size**2 * g_norm / l_min:.2e}")

                ls_core = least_squares(least_squares_sample, least_squares_weights, least_squares_gradient, core_basis_l2)
                assert np.all(np.isfinite(ls_core))
                tt.core -= step_size * ls_core.reshape(tt.core.shape)

                full_sample.extend(least_squares_sample)
                full_weights.extend(least_squares_weights)
                full_values.extend(least_squares_values)
            elif lsid == "CRLS":

                def l2_suboptimality(sample, weights):
                    sample = np.asarray(sample)
                    weights = np.asarray(weights)
                    if len(sample) < core_basis_l2.dimension:
                        return np.inf
                    B = core_basis_l2(sample)
                    assert B.shape == (core_basis_l2.dimension, len(sample))
                    G = B * weights @ B.T / len(sample)
                    assert np.all(0 <= weights)
                    return np.sqrt(1 + np.max(weights) ** 2 / np.linalg.norm(G, ord=-2))

                current_suboptimality = l2_suboptimality(full_sample, full_weights)
                print(f"Suboptimality factor: {current_suboptimality:.1f} < {target_suboptimality:.1f}")
                if current_suboptimality > target_suboptimality:
                    # suboptimality = sqrt(1 + mu^2 tau^2)
                    # --> mu^2 = (suboptimality^2 - 1) / tau^2 --> lambda = 1 / mu^2
                    # current_suboptimality = suboptimality(full_sample)
                    full_sample_test, full_weights_test, full_values_test = full_sample, full_weights, full_values
                    full_sample, full_weights = ensure_l2_stability(
                        rng, core_basis_l2, discretisation, target_suboptimality, full_sample, full_weights
                    )
                    full_sample = list(full_sample)
                    full_weights = list(full_weights)
                    full_values.extend(target(full_sample[len(full_values) :]))
                    assert len(full_sample) == len(full_weights) == len(full_values)
                    assert np.all(np.asarray(full_sample[:len(full_sample_test)]) == full_sample_test)
                    assert np.all(np.asarray(full_weights[:len(full_weights_test)]) == full_weights_test)
                    assert np.all(np.asarray(full_values[:len(full_values_test)]) == full_values_test)
                    current_suboptimality = l2_suboptimality(full_sample, full_weights)
                    print(f"Suboptimality factor: {current_suboptimality:.1f} < {target_suboptimality:.1f}")

                full_gradient = gradient(full_sample, full_values)
                assert np.all(np.isfinite(full_gradient))
                assert np.allclose(
                    evaluate(tt, [l2_basis] * tt.order, test_sample), tt.core.reshape(-1) @ core_basis_l2(test_sample)
                )
                # print(f"  Least squares variance: {np.sqrt(1 + current_suboptimality) * core_basis_l2.dimension:.2e}")

                ls_core = least_squares(full_sample, full_weights, full_gradient, core_basis_l2)
                assert np.all(np.isfinite(ls_core))
                tt.core -= step_size * ls_core.reshape(tt.core.shape)
            elif lsid == "LSSS":
                raise NotImplementedError()
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

                # ls_probability = min(target_suboptimality / l2_suboptimality(full_sample, full_weights), 1)
                # print(f"  Least squares probability: {ls_probability:.2f}")
                # ls_scaling = rng.binomial(1, ls_probability)
                # update_core = ls_scaling * least_squares(full_sample, full_weights, full_grad, core_basis_l2)
                # ls_scaling = min(target_suboptimality / l2_suboptimality(full_sample, full_weights), 1)

                # ls_scaling = 1 / np.sqrt(1 + l2_suboptimality(full_sample, full_weights))
                # print(f"  Least squares scaling: {ls_scaling:.2f}")
                # update_core = ls_scaling * least_squares(full_sample, full_weights, full_grad, core_basis_l2)
            elif lsid == "NONE":
                ls_core = np.zeros(tt.core.size, dtype=float)
            else:
                raise RuntimeError(f"Unknown least squares identifier '{lsid}'")

            print(
                f"Sample size: {len(full_sample)}"
                f"  |  Oversampling: {len(full_sample) / tt.parameters:.2f}"
            )
            test_error = compute_test_error(tt)
            print(f"Relative test set error: {test_error:.2e}")

            # TODO...
            # Although the Christoffel density is optimal for debiasing,
            # it is advantageous to draw from the mixture of the Christoffel density and 1
            # to ensure that full_weights <= 2. To retain the same variance, we have to adapt the step size.
            # Recall that the variance of the quasi-projection is given by k / n, where n is the sample size and
            #     k = max(abs(w * K)) ,
            # with K denoting the unnormalised (inverse) Christoffel function and w the weight function.
            # Sampling from the mixture yields
            #     w = 2 / (1 + K/d) ,
            # where d is the dimension of the space. Therefore, the variance is bounded by
            #     k = max(abs( 2K / (1 + K/d) )) = 2d max(abs( K / (d + K) )) <= 1 .
            # if use_debiasing and (not initialisation_sweep or sweeps > 1):
            #     # If we don't use debiasing, then we have no bound for the variance.
            #     step_size *= min(debiasing_sample_size / (2 * core_basis_l2.dimension), 1)
            #     print(step_size)

            if qpid in ["QP", "QPD"]:
                print("Performing quasi-projection debiasing update...")
                if qpid == "QP":
                    debiasing_sample_size = 1
                else:
                    debiasing_sample_size = core_basis_l2.dimension
                print(f"  Sample size: {debiasing_sample_size}")
                debiasing_sample = core_basis_l2.core_space.christoffel_sample(
                    rng,
                    core_basis_l2.bases,
                    [rho] * order,
                    [discretisation] * order,
                    sample_size=debiasing_sample_size,
                    stratified=True,
                )
                assert space in ["h1", "h10"]
                mask = rng.binomial(1, 0.5, size=len(debiasing_sample)).astype(bool)
                debiasing_sample[mask] = rng.uniform(*l2_basis.domain, size=(np.count_nonzero(mask), order))
                christoffel = lambda points: core_basis_l2.core_space.christoffel(points, core_basis_l2.bases)
                debiasing_weights = 2 / (christoffel(debiasing_sample) + 1)
                assert np.all(debiasing_weights <= 2)
                assert debiasing_weights.shape == (len(debiasing_sample),)
                debiasing_values = target(debiasing_sample)
                debiasing_gradient = gradient(debiasing_sample, debiasing_values)
                assert np.all(np.isfinite(debiasing_gradient))
                projected_debiasing_gradient = ls_core @ core_basis_l2(debiasing_sample)
                qp_core = quasi_projection(
                    debiasing_sample, debiasing_gradient - projected_debiasing_gradient, debiasing_weights, core_basis_l2
                )
                # WAIT A SECOND:
                # Pg + Q(I-P)g = Pg + Qg - QPg = (P-QP)g + Qg = (I-Q)Pg + Qg
                # Thats so shitty!
                # You basically have an entire quasi-projection in there!
                Moreover, if Q is a projection(like a DPP projection), then the debiased gradient is just Q!
                Pg + Qg - QPg = Qg
                assert np.all(np.isfinite(qp_core))
                tt.core -= step_size * qp_core.reshape(tt.core.shape)

                full_sample.extend(debiasing_sample)
                full_weights.extend(debiasing_weights)
                full_values.extend(debiasing_values)
            elif qpid == "NONE":
                pass
            else:
                raise RuntimeError(f"Unknown quasi-projection identifier '{qpid}'")

            print(
                f"Sample size: {len(full_sample)}"
                f"  |  Oversampling: {len(full_sample) / tt.parameters:.2f}"
            )
            test_error = compute_test_error(tt)
            print(f"Relative test set error: {test_error:.2e}")

            errors.append(test_error)
            sample_sizes.append(len(full_sample))

            svs = tt.singular_values()
            erank_ps = lambda s: s / np.sum(s)
            erank = lambda s: np.exp(-np.sum(erank_ps(s) * np.log(erank_ps(s))))
            erks = "[" + ", ".join(f"{erank(svs):.2f}" for svs in svs) + "]"
            print(f"Effective ranks: {erks}")

            if len(full_sample) / tt.parameters > 1000:
                break

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
    plot_path = f"als_{order}_{target.__name__}_{scheme_str}"
    plot_path += ("__" if suffix else "") + "_".join(sorted(suffix))
    plot_path = plot_directory / f"{plot_path}.png"
    print("Saving sample statistics plot to", plot_path)
    plt.savefig(plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True)
    plt.close(fig)
