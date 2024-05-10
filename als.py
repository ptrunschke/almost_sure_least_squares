from __future__ import annotations
from typing import Optional
import numpy as np
from rkhs_1d import H10Kernel, H1Kernel, H1GaussKernel, Kernel, kernel_matrix
from basis_1d import compute_discrete_gramian, enforce_zero_trace, orthonormalise, MonomialBasis, FourierBasis, SinBasis, TransformedBasis, Basis
from sampling import draw_sample, draw_sequence, Sampler
from greedy_subsampling import eta_metric, lambda_metric, greedy_step, Metric
from least_squares import optimal_least_squares

rng = np.random.default_rng(0)

# dimension = 5
dimension = 10
basis_name = "polynomial"
# basis_name = "fourier"
space = ["h10", "h1", "h1gauss"][1]

target_mu = 50
debiasing_sample_size = 1
max_iteration = 500
rank = 4

rank = 6  # just to test the 1 / (1 + sum) test function --- Anthony's experiment

# def target(points: np.ndarray) -> np.ndarray:
#     # Corner peak in two dimensions
#     cs = np.array([3.0, 5.0])
#     points = np.asarray(points)
#     assert points.ndim == 2 and points.shape[1] == len(cs)
#     assert np.allclose(np.asarray(rkhs_basis.domain), [-1, 1])
#     points = (points + 1) / 2  # transform points to interval [0, 1]
#     return (1 + points @ cs)**(-(len(cs) + 1))

def target(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    points = (points + 1) / 2  # transform points to interval [0, 1]
    # return points[:, 1] / (points[:, 0] + 1e-3)
    # return np.cos(points[:, 0] + points[:, 1]) * np.exp(points[:, 0] * points[:, 1])
    return 1 / (1 + np.sum(points, axis=1))


def greedy_bound(selection_metric: Metric, target_metric: Metric, target: float, full_sample: np.ndarray, selected: Optional[list[int]] = None) -> tuple[list[int], float]:
    print("Subsampling...")
    value = -np.inf
    if selected is None:
        selected = []
    else:
        selected = list(selected)
    while len(selected) < len(full_sample) and value < target:
        # for _ in range(core_basis.dimension):
        selected.append(greedy_step(selection_metric, full_sample, selected))
        value = target_metric(full_sample[selected])
        print(f"  Sample size: {len(selected)}  |  Target metric: {value:.2e} < {target:.2e}")
    return selected, value


def create_core_space_sampler(rng: np.random.Generator, core_basis: CoreBasis, discretisation: np.ndarray) -> Sampler:
    rank = core_basis.tt[0].shape[1]

    uni_basis = core_basis.bases[core_basis.core_position]
    uni_density = lambda idx: lambda x: uni_basis(x)[idx]**2 * rho(x)

    red_transform = core_basis.tt[1] if core_basis.core_position == 0 else core_basis.tt[0].T
    red_basis = TransformedBasis(red_transform, core_basis.bases[1 - core_basis.core_position])
    red_density = lambda idx: lambda x: red_basis(x)[idx]**2 * rho(x)

    print("TODO: The quasi-projection is erronous. It is missing the weights.")
    def core_space_sampler(conditioned_on: Optional[list[float]] = None) -> float:
        uni_idx = rng.integers(0, core_basis.bases[core_basis.core_position].dimension)
        uni_sample = draw_sample(rng=rng, density=uni_density(uni_idx), discretisation=discretisation)
        red_idx = rng.integers(0, rank)
        red_sample = draw_sample(rng=rng, density=red_density(red_idx), discretisation=discretisation)

        # weights = core_basis.dimension * rho(uni_sample) * rho(red_sample) / (uni_density(uni_idx)(uni_sample) * red_density(red_idx)(red_sample))

        if core_basis.core_position == 0:
            return np.array([uni_sample, red_sample])
        else:
            return np.array([red_sample, uni_sample])
    return core_space_sampler


if space == "h10":
    def rho(x):
        return np.full(len(x), 0.5)
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
        return np.full(len(x), 0.5)
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
        return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    rkhs_kernel = H1GaussKernel((-8, 8))
    if basis_name == "polynomial":
        initial_basis = MonomialBasis(dimension, domain=(-8, 8))
    elif basis_name == "fourier":
        initial_basis = FourierBasis(dimension, domain=(-8, 8))
    else:
        raise NotImplementedError()
else:
    raise NotImplementedError()

discretisation = np.linspace(*initial_basis.domain, 1000)

discrete_rkhs_gramian = compute_discrete_gramian(space, initial_basis.domain, 2 ** 13)
rkhs_basis = orthonormalise(initial_basis, *discrete_rkhs_gramian)

target_lambda = 1 / target_mu**2  # mu = lambda points: 1 / np.sqrt(lambda_(points))
lambda_to_mu = lambda l: 1 / np.sqrt(l)



def quasi_projection(points: np.ndarray, values: np.ndarray, basis: Basis):
    assert points.ndim >= 1 and values.ndim == 1
    assert len(points) == len(values)
    dimension = np.reshape(basis.domain, (-1, 2)).shape[0]
    if points.ndim == 1 and dimension != 1:
        raise ValueError(f"dimension mismatch: input dimension is 1 but space dimension is {dimension}")
    elif points.shape[1] != dimension:
        raise ValueError(f"dimension mismatch: input dimension is {points.shape[1]} but space dimension is {dimension}")
    return basis(points) @ values / len(points)


TensorTrain = list[np.ndarray]
tt: TensorTrain = [rng.standard_normal(size=(dimension, rank)), rng.standard_normal(size=(rank, dimension))]

def move_core(tt: TensorTrain, position: int):
    if position == 0:
        u, s, vt = np.linalg.svd(tt[1])
        u, s, vt = u[:, :rank], s[:rank], vt[:rank]
        tt[0] = tt[0] @ u * s
        tt[1] = vt
    elif position == 1:
        u, s, vt = np.linalg.svd(tt[0])
        u, s, vt = u[:, :rank], s[:rank], vt[:rank]
        tt[0] = u
        tt[1] = s[:, None] * vt @ tt[1]
    else:
        raise ValueError(f"Invalid position '{position}'")

def evaluate(tt: TensorTrain, points: np.ndarray) -> np.ndarray:
    assert points.ndim == 2 and points.shape[1] == len(tt)
    assert len(tt) == 2
    lbs = rkhs_basis(points[:, 0])
    assert lbs.shape == (dimension, len(points))
    rbs = rkhs_basis(points[:, 1])
    assert rbs.shape == (dimension, len(points))
    return np.einsum("dr,re,di,ei -> i", *tt, lbs, rbs)


class CoreBasis(Basis):
    def __init__(self, tt: TensorTrain, bases: list[Basis], core_position: int):
        assert len(tt) == len(bases)
        assert 0 <= core_position < len(tt)
        self.tt = tt
        self.bases = bases
        self.core_position = core_position

    @property
    def dimension(self) -> int:
        assert len(self.tt) == 2
        assert self.tt[0].shape[1] == self.tt[1].shape[0]
        rank = self.tt[0].shape[1]
        dim = self.tt[0].shape[0] if self.core_position == 0 else self.tt[1].shape[1]
        return dim * rank

    @property
    def domain(self) -> list[tuple[float, float]]:
        return [b.domain for b in self.bases]

    def __call__(self, points: np.ndarray) -> np.ndarray:
        assert points.ndim == 2 and points.shape[1] == len(self.tt)
        assert len(tt) == 2
        bs = [b(points[:, m]) for m, b in enumerate(self.bases)]
        if self.core_position == 0:
            return np.einsum("re,di,ei -> dri", self.tt[1], *bs).reshape(self.dimension, len(points))
        else:
            return np.einsum("dr,di,ei -> eri", self.tt[0], *bs).reshape(self.dimension, len(points))


class TensorProductKernel(Kernel):
    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    @property
    def domain(self) -> list[tuple[float, float]]:
        return [k.domain for k in self.kernels]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert x.ndim >= 2 and y.ndim >= 2 and x.shape[-1] == len(self.kernels)
        assert x.shape[-1] == y.shape[-1]
        res = 1
        for i, k in enumerate(self.kernels):
            res *= k(x[..., i], y[..., i])
        return res


product_kernel = TensorProductKernel(rkhs_kernel, rkhs_kernel)


def ensure_stability(core_basis: CoreBasis, target_metric: Metric, target: float) -> np.ndarray:
    # Draw new samples until the target is over-satisfied by a factor of 1/(1-1/e) = e / (e - 1).
    print("Ensuring stability...")
    core_space_sampler = create_core_space_sampler(rng, core_basis, discretisation)
    candidates = []
    target = np.e / (np.e - 1) * target
    value = -np.inf
    while value < target:
        candidates.extend(draw_sequence(core_space_sampler, core_basis.dimension))
        value = target_metric(candidates)
        print(f"  Sample size: {len(candidates)}  |  Target metric: {value:.2e} < {target:.2e}")
    candidates = np.asarray(candidates)
    assert candidates.ndim == 2 and candidates.shape[1] == 2
    return candidates


def greedy_draw(selection_metric: Metric, selection_size: int, full_sample: np.ndarray, selected: Optional[list[int]] = None) -> list[int]:
    print("Subsampling...")
    if selected is None:
        selected = []
    else:
        selected = list(selected)
    while len(selected) < selection_size:
        selected.append(greedy_step(selection_metric, full_sample, selected))
    return selected


if space in ["h1", "h10"]:
    test_sample = rng.uniform(-1, 1, size=(10000, 2))
elif space == "h1gauss":
    test_sample = rng.standard_normal(size=(10000, 2))
else:
    raise NotImplementedError()
test_values = target(test_sample)

full_sample = np.empty(shape=(0, 2), dtype=float)
full_values = np.empty(shape=(0,), dtype=float)
core_position = -1
errors = []
sample_sizes = []
for it in range(max_iteration):
    core_position = (core_position + 1) % 2
    move_core(tt, core_position)
    step_size = 1 / np.sqrt(it + 1)
    # step_size = 1 / (it + 1)
    # step_size = 5e-2
    # step_size = 1
    print(f"Iteration: {it}  |  Core position: {core_position}  |  Step size: {step_size:.2e}")

    core_basis = CoreBasis(tt, [rkhs_basis]*2, core_position)
    assert len(tt) == 2
    if core_position == 0:
        core = tt[0].reshape(-1)
    else:
        core = tt[1].T.reshape(-1)
    assert np.allclose(core_basis(test_sample).T @ core, evaluate(tt, test_sample))
    test_error = core_basis(test_sample).T @ core - test_values
    assert test_error.shape == test_values.shape
    test_error = np.linalg.norm(test_error, ord=2) / np.linalg.norm(test_values, ord=2)
    print(f"Relative test set error: {test_error:.2e}")
    errors.append(test_error)
    sample_sizes.append(len(full_sample))

    eta = eta_metric(product_kernel, core_basis)
    lambda_ = lambda_metric(product_kernel, core_basis)

    def suboptimality(points: np.ndarray) -> float:
        points = np.asarray(points)
        assert points.ndim >= 1
        K = product_kernel(points[:, None], points[None, :])
        assert K.shape == (len(points), len(points))
        M = core_basis(points)
        assert M.shape == (core_basis.dimension, len(points))
        # We first compute G = bs @ inv(K) @ bs.T.
        G = np.linalg.lstsq(K, M.T, rcond=None)[0]
        assert G.shape == (len(points), core_basis.dimension)
        G = M @ G
        mu_squared = 1 / min(max(np.linalg.norm(G, ord=-2), 0), 1)
        I = np.eye(core_basis.dimension)
        tau = min(np.linalg.norm(G - I, ord=2) + np.linalg.norm(G - I, ord="fro"), 1)
        # tau = min(2 * np.linalg.norm(G - I, ord="fro"), 1)
        return np.sqrt(1 + mu_squared * tau**2)

    current_lambda = lambda_(full_sample) if len(full_sample) > 0 else 0
    print(f"Initial sample size: {len(full_sample)}  |  Initial mu: {lambda_to_mu(current_lambda):.1f} < {target_mu:.1f}")
    print(f"Initial suboptimality factor: {suboptimality(full_sample):.1f}")
    draw_for_stability = True
    if draw_for_stability:
        if lambda_to_mu(current_lambda) > target_mu:
            candidates = ensure_stability(core_basis, lambda_, target_lambda)
            candidates = np.concatenate([full_sample, candidates], axis=0)
            selected = list(range(len(full_sample)))
            selection_size = max(len(full_sample) + 2, 2 * core_basis.dimension)
            selected = greedy_draw(eta, selection_size, candidates, selected)
            assert np.all(np.asarray(selected[:len(full_sample)]) == np.arange(len(full_sample)))
            new_selected = selected[len(full_sample):]
            full_values = np.concatenate([full_values, target(candidates[new_selected])], axis=0)
            full_sample = candidates[selected]
            selected_lambda = lambda_(full_sample)
            # candidates = ensure_stability(core_basis, lambda_, target_lambda)
            # candidates = np.concatenate([full_sample, candidates], axis=0)
            # selected = list(range(len(full_sample)))
            # selected, selected_lambda = greedy_bound(eta, lambda_, target_lambda, candidates, selected)
            # assert np.all(np.asarray(selected[:len(full_sample)]) == np.arange(len(full_sample)))
            # new_selected = selected[len(full_sample):]
            # full_values = np.concatenate([full_values, target(candidates[new_selected])], axis=0)
            # full_sample = candidates[selected]
        else:
            selected_lambda = current_lambda
        assert full_values.ndim == 1 and full_sample.shape == (len(full_values), 2)
        print(f"Sample size: {len(full_sample)}  |  mu: {lambda_to_mu(selected_lambda):.1f} < {target_mu:.1f}")
        print(f"Suboptimality factor: {suboptimality(full_sample):.1f}")
    else:
        print("Skipping stability check...")

    K = kernel_matrix(product_kernel, full_sample)
    es = np.linalg.eigvalsh(K)
    while np.any(es < 1e-8):
        print("Warning: Kernel matrix is not positive definite")
        print("Removing duplicates...")
        ds = np.linalg.norm(full_sample[:, None] - full_sample[None, :], axis=2)
        assert np.all(ds >= 0) and np.all(ds == ds.T)
        js, ks = np.where(ds < 1e-8)
        js = np.max(js[js != ks])
        print(f"  Removing sample {js+1} / {len(full_sample)}")
        full_sample = np.delete(full_sample, js, axis=0)
        full_values = np.delete(full_values, js, axis=0)
        K = kernel_matrix(product_kernel, full_sample)
        es = np.linalg.eigvalsh(K)

    print("Draw debiasing sample...")
    print("TODO: This sample should be drawn according to an L2 basis!")
    print("TODO: Note that the 'draw_for_stability' may be misleadingly bad, just because it relies heavier on the quasi-projection which suffers from at least two errors!")
    debiasing_sample = draw_sequence(create_core_space_sampler(rng, core_basis, discretisation), debiasing_sample_size)
    debiasing_sample = np.asarray(debiasing_sample)
    debiasing_values = target(debiasing_sample)

    def gradient(points: np.ndarray, values: np.ndarray) -> np.ndarray:
        # L(v) = 0.5 |v - u|^2  -->  grad(L(v)) = v - u
        # v - grad(L(v)) = v - (v - u) = u ✅
        assert values.ndim == 1 and points.shape == values.shape + (2,)
        prediction = core_basis(points).T @ core
        assert prediction.shape == values.shape
        return prediction - values

    update_in_stable_space = False
    use_stable_projection = True
    if update_in_stable_space:
        print("Note that this is a generalisation of the conditionally stable projector from Cohen and Migliorati.")
        print("They use the empirical projector only if it is stable. We use the stable subspace, which may be zero-dimensional.")
        print("One problem is, that we don't have any guarantees about |(I-P) grad| <= |P grad|.")
        print("This means that even though the update is stable, it may be detrimental.")
        print("Computing stable directions...")
        stability_threshold = 10
        print(f"Stability threshold: {stability_threshold:.1f}")
        K = product_kernel(full_sample[:, None], full_sample[None, :])
        assert K.shape == (len(full_sample), len(full_sample))
        M = core_basis(full_sample)
        assert M.shape == (core_basis.dimension, len(full_sample))
        # We want to compute M @ inv(K) @ M.T
        G = np.linalg.lstsq(K, M.T, rcond=None)[0]
        assert G.shape == (len(full_sample), core_basis.dimension)
        G = M @ G
        es, vs = np.linalg.eigh(G)
        assert np.allclose(vs * es @ vs.T, G)
        es = np.maximum(es, 0)
        # The stability condition is (1 + mu^2 tau^2)^0.5 <= stability_threshold with
        #     mu^2 = 1 / lambda_min(G) and
        #     tau^2 = min(4 ||G - I||_F^2, 1) .
        # It holds that
        #     mu^2 = 1 / min(es) 
        #     tau^2 = min(4 sum((es - 1)^2), 1) .
        stab = lambda es: np.sqrt(1 + min(4 * np.sum((es - 1)**2), 1) / np.min(es))
        indices = np.arange(len(es))
        mask = np.full(len(es), True)
        while np.any(mask) and stab(es[mask]) > stability_threshold:
            rm_idx = np.argmin(es[mask])
            mask[indices[mask][rm_idx]] = False
        # P = vs[:, mask] @ vs[:, mask].T
        stable_space_dimension = np.count_nonzero(mask)
        P = vs[:, mask].T
        stable_basis = TransformedBasis(P, core_basis)
        print(f"Stable space dimension: {stable_space_dimension} / {core_basis.dimension}")

    full_grad = gradient(full_sample, full_values)
    assert np.all(np.isfinite(full_grad))
    if update_in_stable_space:
        update_core = P.T @ optimal_least_squares(full_sample, full_grad, product_kernel, stable_basis)
    elif use_stable_projection:
        update_core = optimal_least_squares(full_sample, full_grad, product_kernel, core_basis)
        if suboptimality(full_sample) > target_mu:
            update_core *= 0
    else:
        update_core = optimal_least_squares(full_sample, full_grad, product_kernel, core_basis)
    assert np.all(np.isfinite(update_core))
    if (not draw_for_stability) or lambda_to_mu(selected_lambda) >= 5:
        debiasing_grad = gradient(debiasing_sample, debiasing_values)
        assert np.all(np.isfinite(debiasing_grad))
        residual = debiasing_grad - core_basis(debiasing_sample).T @ update_core
        assert np.all(np.isfinite(residual))
        debiasing_core = quasi_projection(debiasing_sample, residual, core_basis)
        assert np.all(np.isfinite(debiasing_core))
        update_core += debiasing_core
    elif update_in_stable_space:
        assert stable_space_dimension > 0
    assert len(tt) == 2
    if core_position == 0:
        update_core = update_core.reshape(dimension, rank)
    else:
        update_core = update_core.reshape(dimension, rank).T
    tt[core_position] -= step_size * update_core

    if (not draw_for_stability) or lambda_to_mu(selected_lambda) >= 5:
        full_sample = np.concatenate([full_sample, debiasing_sample], axis=0)
        full_values = np.concatenate([full_values, debiasing_values], axis=0)

try:
    import matplotlib.pyplot as plt
    plt.plot(np.asarray(sample_sizes) / sum(cmp.size for cmp in tt), errors)
    plt.yscale("log")
    plt.xlabel("sample size / number of parameters")
    plt.show()
finally:
    from IPython import embed
    # embed()
