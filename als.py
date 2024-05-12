from __future__ import annotations
from typing import Optional
from jaxtyping import Float
import numpy as np
from opt_einsum import contract
from basis_1d import TransformedBasis, Basis
from sampling import draw_weighted_sequence, Sampler
from greedy_subsampling import fast_greedy_step, Metric, FastMetric
from tensor_train import TensorTrain, TensorTrainCoreSpace


def ensure_stability(core_basis: CoreBasis, target_metric: Metric, target: float, repetitions: int = 10) -> np.ndarray:
    # Draw new samples until the target is over-satisfied by a factor of 1/(1-1/e) = e / (e - 1).
    print("Ensuring stability...")
    core_space_sampler = create_core_space_sampler(rng, core_basis, discretisation)
    candidates = []
    target = np.e / (np.e - 1) * target
    value = -np.inf
    while value < target:
        for _ in range(repetitions):
            extension = draw_weighted_sequence(core_space_sampler, core_basis.dimension)[0]
            extended_value = target_metric(candidates + extension)
            if extended_value > value:
                extended_candidates = candidates + extension
                value = extended_value
        candidates = extended_candidates
        print(f"  Sample size: {len(candidates)}  |  Target metric: {value:.2e} < {target:.2e}")
    candidates = np.asarray(candidates)
    assert candidates.ndim == 2 and candidates.shape[1] == 2
    return candidates


def greedy_bound(selection_metric_factory: FastMetric, target_metric: Metric, target: float, full_sample: np.ndarray, selected: Optional[list[bool]] = None) -> list[bool]:
    print("Subsampling...")
    if selected is None:
        selected = np.full(len(full_sample), False, dtype=bool)
    else:
        selected = np.array(selected)
    indices = np.arange(len(full_sample))
    value = -np.inf
    for selection_size in range(np.count_nonzero(selected), len(full_sample)):
        if value >= target:
            break
        selection_metric = selection_metric_factory(full_sample[selected])
        selected[indices[~selected][fast_greedy_step(selection_metric, full_sample[~selected])]] = True
        value = target_metric(full_sample[selected])
        print(f"  Sample size: {selection_size}  |  Target metric: {value:.2e} < {target:.2e}")
    return selected


def greedy_draw(selection_metric_factory: FastMetric, selection_size: int, full_sample: np.ndarray, selected: Optional[list[bool]] = None) -> list[bool]:
    print("Subsampling...")
    if selected is None:
        selected = np.full(len(full_sample), False, dtype=bool)
    else:
        selected = np.array(selected)
    indices = np.arange(len(full_sample))
    for _ in range(selection_size - np.count_nonzero(selected)):
        selection_metric = selection_metric_factory(full_sample[selected])
        selected[indices[~selected][fast_greedy_step(selection_metric, full_sample[~selected])]] = True
    return selected


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

    def __call__(self, points: Float[np.ndarray, "sample_size dimension"]) -> Float[np.ndarray, "core_dimension sample_size"]:
        assert points.ndim == 2 and points.shape[1] == self.core_space.tensor_train.order
        bs = [b(points[:, m]) for m, b in enumerate(self.bases)]
        ln, en, rn = self.core_space.evaluate(bs)
        return contract("ln, en, rn -> lern", ln, en, rn).reshape(self.dimension, points.shape[0])


def evaluate(tt: TensorTrain, bases: list[Basis], points: Float[np.ndarray, "sample_size dimension"]) -> Float[np.ndarray, "sample_size"]:
    return CoreBasis(tt, bases)(points).T @ tt.core.reshape(-1)


def create_core_space_sampler(rng: np.random.Generator, core_basis: CoreBasis, discretisation: Float[np.ndarray, "discretisation"]) -> Sampler:
    core_space = core_basis.core_space
    order = core_space.tensor_train.order
    def core_space_sampler(conditioned_on : Optional[tuple[list[Float[np.ndarray, "dimension"]], list[float]]] = None) -> tuple[Float[np.ndarray, "dimension"], float]:
        sample = core_space.christoffel_sample(rng, core_basis.bases, [rho] * order, [discretisation] * order)
        sample = np.array(sample, dtype=float)
        weight = 1 / core_space.christoffel(sample, core_basis.bases)
        return sample, weight
    return core_space_sampler


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

    from rkhs_1d import H10Kernel, H1Kernel, H1GaussKernel, kernel_matrix
    from rkhs_nd import TensorProductKernel
    from basis_1d import compute_discrete_gramian, enforce_zero_trace, orthonormalise, orthogonalise, MonomialBasis, FourierBasis, SinBasis
    from greedy_subsampling import fast_eta_metric, lambda_metric, suboptimality_metric
    from least_squares import optimal_least_squares

    rng = np.random.default_rng(0)
    postfix = set()

    # dimension = 5
    dimension = 10
    basis_name = "polynomial"
    # basis_name = "fourier"
    space = ["h10", "h1", "h1gauss"][1]

    target_suboptimality = 50
    # target_suboptimality = 5; prefix = "ts5"
    debiasing_sample_size = 1
    max_iteration = 2000


    all_parameters = {"draw_for_stability", "draw_for_stability_bound", "update_in_stable_space", "use_stable_projection", "use_debiasing"}

    # Original algorithm
    # used_parameters = {"draw_for_stability", "use_debiasing"}
    used_parameters = {"draw_for_stability_bound", "use_debiasing"}

    # # Try to use fewer samples by updating only in the subspace where G(x) is stable.
    # #     Note that this is a generalisation of the conditionally stable projector from Cohen and Migliorati.
    # #     They use the empirical projector only if it is stable. We use the stable subspace, which may be zero-dimensional.
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

    assert draw_for_stability or draw_for_stability_bound or use_debiasing


    order = 2
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
    product_kernel = TensorProductKernel(rkhs_kernel, rkhs_kernel)

    discretisation = np.linspace(*initial_basis.domain, 1000)

    # if space == "h1gauss":
    #     discrete_l2_gramian = compute_discrete_gramian("l2gauss", initial_basis.domain, 2 ** 13)
    # else:
    #     discrete_l2_gramian = compute_discrete_gramian("l2", initial_basis.domain, 2 ** 13)
    # l2_basis = orthonormalise(initial_basis, *discrete_l2_gramian)

    # discrete_rkhs_gramian = compute_discrete_gramian(space, initial_basis.domain, 2 ** 13)
    # rkhs_basis = orthonormalise(initial_basis, *discrete_rkhs_gramian)

    discrete_rkhs_gramian = compute_discrete_gramian(space, initial_basis.domain, 2 ** 13)
    rkhs_basis = orthonormalise(initial_basis, *discrete_rkhs_gramian)
    if space == "h1gauss":
        discrete_l2_gramian = compute_discrete_gramian("l2gauss", rkhs_basis.domain, 2 ** 13)
    else:
        discrete_l2_gramian = compute_discrete_gramian("l2", rkhs_basis.domain, 2 ** 13)
    rkhs_basis, l2_norms = orthogonalise(rkhs_basis, *discrete_l2_gramian)
    l2_normalise = np.diag(1 / l2_norms)
    l2_basis = TransformedBasis(l2_normalise, rkhs_basis)


    if space in ["h1", "h10"]:
        test_sample = rng.uniform(-1, 1, size=(10000, 2))
    elif space == "h1gauss":
        test_sample = rng.standard_normal(size=(10000, 2))
    else:
        raise NotImplementedError()
    test_values = target(test_sample)

    def compute_test_error(tt: TensorTrain) -> float:
        prediction = evaluate(tt, [rkhs_basis]*tt.order, test_sample)
        assert prediction.shape == test_values.shape
        return np.linalg.norm(prediction - test_values, ord=2) / np.linalg.norm(test_values, ord=2)

    plt.style.use('classic')
    fig, ax = plt.subplots(1, 1)
    # fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    for rank in ranks:
        print(f"Initialise TT of rank {rank}")
        tt = TensorTrain.random(rng, [dimension]*order, [1] + [rank]*(order-1) + [1])

        full_sample = np.empty(shape=(0, 2), dtype=float)
        full_values = np.empty(shape=(0,), dtype=float)
        core_position = -1
        errors = []
        sample_sizes = []
        assert tt.core_position == 0
        all_directions = {0: "right", tt.order - 1: "left"}
        for it in range(max_iteration):
            if tt.core_position in all_directions:
                direction = all_directions[tt.core_position]
            tt.move_core(direction)
            step_size = 1 / np.sqrt(it + 1)
            # step_size = 1; postfix |= {"ss1",}
            print(f"Iteration: {it}  |  Core position: {tt.core_position}  |  Step size: {step_size:.2e}")

            test_error = compute_test_error(tt)
            errors.append(test_error)
            sample_sizes.append(len(full_sample))
            print(f"Relative test set error: {test_error:.2e}")

            core_basis_rkhs = CoreBasis(tt, [rkhs_basis]*2)
            eta_factory = lambda points: fast_eta_metric(product_kernel, core_basis_rkhs, points)
            lambda_ = lambda_metric(product_kernel, core_basis_rkhs)
            suboptimality = suboptimality_metric(product_kernel, core_basis_rkhs)

            current_suboptimality = suboptimality(full_sample)
            print(f"Sample size: {len(full_sample)}  |  Oversampling: {len(full_sample) / tt.parameters:.2f}  |  Suboptimality factor: {current_suboptimality:.1f} < {target_suboptimality:.1f}")
            if draw_for_stability or draw_for_stability_bound:
                if current_suboptimality > target_suboptimality:
                    # suboptimality = sqrt(1 + mu^2 tau^2) --> mu^2 = (suboptimality^2 - 1) / tau^2 --> lambda = 1 / mu^2
                    candidates = ensure_stability(core_basis_rkhs, lambda_, 1 / (target_suboptimality**2 - 1))
                    candidates = np.concatenate([full_sample, candidates], axis=0)
                    selected = np.full(len(candidates), False, dtype=bool)
                    selected[:len(full_sample)] = True
                    selection_size = max(len(full_sample) + 2, 2 * core_basis_rkhs.dimension)
                    if draw_for_stability:
                        selected = greedy_draw(eta_factory, selection_size, candidates, selected)
                    else:
                        assert draw_for_stability_bound
                        selected = greedy_bound(eta_factory, lambda_, 1 / (target_suboptimality**2 - 1), candidates, selected)
                    assert np.all(selected[:len(full_sample)])
                    new_selected = selected[len(full_sample):]
                    full_values = np.concatenate([full_values, target(candidates[len(full_sample):][new_selected])], axis=0)
                    full_sample = candidates[selected]
                    assert full_values.ndim == 1 and full_sample.shape == (len(full_values), 2)
                    current_suboptimality = suboptimality(full_sample)
                    print(f"Sample size: {len(full_sample)}  |  Oversampling: {len(full_sample) / tt.parameters:.2f}  |  Suboptimality factor: {current_suboptimality:.1f} < {target_suboptimality:.1f}")

            K = kernel_matrix(product_kernel, full_sample)
            es = np.linalg.eigvalsh(K)
            while np.any(es < 1e-8):
                print("WARNING: kernel matrix is not positive definite")
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

            if update_in_stable_space:
                print("Computing stable directions...")
                stability_threshold = 10
                print(f"Stability threshold: {stability_threshold:.1f}")
                K = product_kernel(full_sample[:, None], full_sample[None, :])
                assert K.shape == (len(full_sample), len(full_sample))
                M = core_basis_rkhs(full_sample)
                assert M.shape == (core_basis_rkhs.dimension, len(full_sample))
                # We want to compute M @ inv(K) @ M.T
                G = np.linalg.lstsq(K, M.T, rcond=None)[0]
                assert G.shape == (len(full_sample), core_basis_rkhs.dimension)
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
                stable_basis = TransformedBasis(P, core_basis_rkhs)
                print(f"Stable space dimension: {stable_space_dimension} / {core_basis_rkhs.dimension}")


            def gradient(points: np.ndarray, values: np.ndarray) -> np.ndarray:
                # L(v) = 0.5 |v - u|^2  -->  grad(L(v)) = v - u
                # v - grad(L(v)) = v - (v - u) = u ✅
                assert values.ndim == 1 and points.shape == values.shape + (2,)
                prediction = evaluate(tt, [rkhs_basis]*2, points)
                assert prediction.shape == values.shape
                return prediction - values


            full_grad = gradient(full_sample, full_values)
            assert np.all(np.isfinite(full_grad))
            if update_in_stable_space:
                update_core = P.T @ optimal_least_squares(full_sample, full_grad, product_kernel, stable_basis)
            elif use_stable_projection:
                if current_suboptimality <= target_suboptimality:
                    update_core = optimal_least_squares(full_sample, full_grad, product_kernel, core_basis_rkhs)
                else:
                    update_core = np.full(tt[core_position].size, fill_value=0.0, dtype=float)
            else:
                update_core = optimal_least_squares(full_sample, full_grad, product_kernel, core_basis_rkhs)
            assert np.all(np.isfinite(update_core))

            if use_debiasing:
                print("Draw debiasing sample...")
                core_basis_l2 = CoreBasis(tt, [l2_basis]*2)
                # TODO: The basis is wrong! We should use a transformed version of tt!
                debiasing_sample, debiasing_weights = draw_weighted_sequence(create_core_space_sampler(rng, core_basis_l2, discretisation), debiasing_sample_size)
                debiasing_sample, debiasing_weights = np.asarray(debiasing_sample), np.asarray(debiasing_weights)
                assert debiasing_weights.shape == (len(debiasing_sample),)
                debiasing_values = target(debiasing_sample)
                debiasing_grad = gradient(debiasing_sample, debiasing_values)
                assert np.all(np.isfinite(debiasing_grad))
                residual = debiasing_grad - core_basis_rkhs(debiasing_sample).T @ update_core
                assert np.all(np.isfinite(residual))
                debiasing_core = quasi_projection(debiasing_sample, residual, debiasing_weights, core_basis_rkhs)
                # TODO: ideally, i would like to compute
                #     quasi_projection(debiasing_sample, residual, debiasing_weights, core_basis_l2)
                #     and then perform a change of basis to core_basis_rkhs
                assert np.all(np.isfinite(debiasing_core))
                update_core += debiasing_core

            tt.core -= step_size * update_core.reshape(tt.core.shape)

            if use_debiasing:
                full_sample = np.concatenate([full_sample, debiasing_sample], axis=0)
                full_values = np.concatenate([full_values, debiasing_values], axis=0)

            max_derivative = np.log(np.max(errors[-50:])) - np.log(np.min(errors[-50:]))
            print(f"Max derivative: {max_derivative:.2e}")
            if len(errors) >= 50 and max_derivative <= 0.05:
                break
            print()

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
        ax.plot(np.asarray(sample_sizes) / tt.parameters, errors, ">--", color=color, fillstyle="none", linewidth=1.5, markeredgewidth=1.5, markersize=6, label=f"$r = {rank}$")
        print()

    ax.set_yscale("log")
    ax.set_xlabel("nb. evaluations / nb. parameters")
    ax.set_ylabel("error")
    ax.legend()

    plot_directory = Path(__file__).parent / "plot"
    plot_directory.mkdir(exist_ok=True)
    plot_file = f"als_{order}_{target.__name__}"
    plot_file += "_" + "_".join(sorted(used_parameters))
    plot_file += ("_" if postfix else "") + "_".join(sorted(postfix))
    plot_path = plot_directory / f"{plot_file}.png"
    print("Saving sample statistics plot to", plot_path)
    plt.savefig(
        plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )
    # plt.show()
    plt.close(fig)
