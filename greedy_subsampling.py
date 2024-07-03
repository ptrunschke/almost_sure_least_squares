from collections.abc import Callable
import numpy as np
from tqdm import trange
from basis_1d import Basis
from rkhs_1d import Kernel


Metric = Callable[[np.ndarray], float]


def lambda_metric(kernel: Kernel, basis: Basis) -> Metric:
    def lambda_(points: np.ndarray) -> float:
        points = np.asarray(points)
        assert points.ndim == 1
        K = kernel(points[:, None], points[None, :])
        bs = basis(points)
        assert K.shape == (len(points), len(points)) and bs.shape == (basis.dimension, len(points))
        # We want to compute lambda(x) = lambda_min(bs @ inv(K) @ bs.T).
        cs = np.linalg.lstsq(K, bs.T, rcond=None)[0]
        assert cs.shape == (len(points), basis.dimension)
        return min(max(np.linalg.norm(bs @ cs, ord=-2), 0), 1)
    return lambda_


def eta_metric(kernel: Kernel, basis: Basis) -> Metric:
    def eta(points: np.ndarray) -> float:
        assert points.ndim == 1
        K = kernel(points[:, None], points[None, :])
        bs = basis(points)
        assert K.shape == (len(points), len(points)) and bs.shape == (basis.dimension, len(points))
        # We want to compute eta(x) = sum_j |P_{\mcal{V}_x} b[j]|_V^2 = sum_j |b[j]|_x^2 = sum_j bs[j] @ inv(K) @ bs[j].
        # For this, we need to compute cs[j] := K(x)^{-1} b[j](x) = inv(K) @ bs[j] = (inv(K) @ bs.T).T[j].
        cs = np.linalg.lstsq(K, bs.T, rcond=None)[0].T
        assert cs.shape == (basis.dimension, len(points))
        # Then eta(x) = sum_j bs[j] @ cs[j] = sum_{i,j} bs[j, i] * cs[j, i].
        return min(max(np.sum(bs * cs), 0), basis.dimension)
    return eta


def greedy_step(metric: Metric, full_sample: np.ndarray, selected: list[int]) -> int:
    assert np.ndim(full_sample) == 1
    assert len(selected) == 0 or (0 <= min(selected) and max(selected) < len(full_sample))
    candidates = np.full(len(full_sample), False)
    candidates[selected] = True
    etas = np.empty(len(full_sample))
    for index in range(len(full_sample)):
        if index in selected:
            etas[index] = -np.inf
        else:
            candidates[index] = True
            etas[index] = metric(full_sample[candidates])
            candidates[index] = False
    opt = np.argmax(etas)
    return opt


if __name__ == "__main__":
    from typing import Optional
    from pathlib import Path
    # from functools import partial

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # from matplotlib import ticker
    from tqdm import trange
    from plotting import plotting

    from basis_1d import create_subspace_kernel
    from sampling import draw_sample

    # class OffsetLogLocator(ticker.LogLocator):
    #     def __init__(self, offset) -> None:
    #         super().__init__()
    #         self.offset = offset

    #     def tick_values(self, vmin: float, vmax: float):
    #         return super().tick_values(vmin - self.offset, vmax - self.offset) + self.offset

    #     def view_limits(self, vmin: float, vmax: float):
    #         lmin, lmax = super().view_limits(vmin - self.offset, vmax - self.offset)
    #         return lmin + 1, lmax + 1

    from rkhs_1d import H10Kernel, H1Kernel, H1GaussKernel
    from basis_1d import compute_discrete_gramian, enforce_zero_trace, orthonormalise, MonomialBasis, FourierBasis, SinBasis
    from sampling import draw_embedding_sample, draw_sequence

    rng = np.random.default_rng(0)

    # dimension = 5
    dimension = 10
    basis_name = "polynomial"
    # basis_name = "fourier"

    full_sample_size = 100
    max_subsample_size = 20
    trials = 100

    tab20 = mpl.colormaps["tab20"].colors
    geometry = {
        "top": 1,
        "bottom": 0,
        "left": 0,
        "right": 1,
        "wspace": 0.25,  # the default as defined in rcParams
        "hspace": 0.25,  # the default as defined in rcParams
    }
    figshape = (1, 1)
    textwidth = 6.50127  # width of the text in inches
    figwidth = textwidth  # width of the figure in inches
    # figwidth = textwidth / 2  # width of the figure in inches
    phi = (1 + np.sqrt(5)) / 2
    aspect_ratio = phi
    figsize = plotting.compute_figsize(geometry, figshape, aspect_ratio=aspect_ratio, figwidth=figwidth)

    for space in ["h10", "h1", "h1gauss"]:
        print(f"Compute subsample statistics for {space} with {basis_name} basis")
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

        discrete_rkhs_gramian = compute_discrete_gramian(space, initial_basis.domain, 2 ** 13)
        rkhs_basis = orthonormalise(initial_basis, *discrete_rkhs_gramian)

        if space == "h1gauss":
            rho = lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
            discrete_l2_gramian = compute_discrete_gramian("l2gauss", initial_basis.domain, 2 ** 13)
        else:
            rho = lambda x: np.full(len(x), 0.5)
            discrete_l2_gramian = compute_discrete_gramian("l2", initial_basis.domain, 2 ** 13)
        l2_basis = orthonormalise(initial_basis, *discrete_l2_gramian)

        # sampler = partial(draw_embedding_sample, rng=rng, rkhs_kernel=rkhs_kernel, rkhs_basis=rkhs_basis, l2_basis=l2_basis, reference_density=rho, discretisation=discretisation)

        subspace_l2_kernel = create_subspace_kernel(l2_basis)
        def christoffel_sampler(conditioned_on: Optional[list[float]] = None) -> float:
            weighted_density = lambda x: subspace_l2_kernel(x, x) * rho(x)
            return draw_sample(rng=rng, density=weighted_density, discretisation=discretisation)

        sampler = christoffel_sampler

        eta = eta_metric(rkhs_kernel, rkhs_basis)
        lambda_ = lambda_metric(rkhs_kernel, rkhs_basis)
        mu = lambda points: 1 / np.sqrt(lambda_(points))

        etas = np.empty((max_subsample_size, trials))
        mus = np.empty((max_subsample_size, trials))
        for trial in trange(trials, position=0):
            full_sample = draw_sequence(sampler, full_sample_size, verbose=False)
            full_sample = np.asarray(full_sample)
            assert full_sample.shape == (full_sample_size,)
            mu_value = mu(full_sample)

            selected = []
            for subsample_size in trange(max_subsample_size, position=1, leave=False):
                selected.append(greedy_step(eta, full_sample, selected))
                etas[subsample_size, trial] = eta(full_sample[selected])
                mus[subsample_size, trial] = mu(full_sample[selected])

        assert np.all(0 < etas) and np.all(etas <= rkhs_basis.dimension)
        assert np.all(1 <= mus)

        fig, ax_1 = plt.subplots(*figshape, figsize=figsize, dpi=300)
        positions = np.arange(1, max_subsample_size+1)
        parts = ax_1.violinplot(etas.T, positions=positions-0.15, widths=0.35)
        for pc in parts['bodies']:
            # pc.set_edgecolor(tab20[0])
            pc.set_facecolor(tab20[1])
            pc.set_alpha(1)
        for key in parts.keys():
            if key == "bodies":
                continue
            parts[key].set_color(tab20[0])
            pc.set_alpha(1)
        ax_1.set_xticks(np.arange(1, max_subsample_size+1, max_subsample_size // 9))
        ax_1.set_xlabel("sample size")
        ax_1.set_yscale("linear")
        ax_1.set_ylim(0, np.ceil(np.max(etas)) + 0.5)
        ax_1.set_ylabel(r"$\eta$", color="tab:blue", rotation=0, labelpad=8)
        ax_1.set_yticks(np.arange(1, int(np.ceil(np.max(etas))) + 1))
        ax_1.set_yticklabels([f"${i}$" for i in range(1, int(np.ceil(np.max(etas))) + 1)])
        ax_1.tick_params(axis='y', which="both", labelcolor="tab:blue")

        ax_2 = ax_1.twinx()
        parts = ax_2.violinplot(mus[dimension-1:].T, positions=positions[dimension-1:]+0.15, widths=0.35)
        for pc in parts["bodies"]:
            # pc.set_edgecolor(tab20[6])
            pc.set_facecolor(tab20[7])
            pc.set_alpha(1)
        for key in parts.keys():
            if key == "bodies":
                continue
            parts[key].set_color(tab20[6])
            pc.set_alpha(1)
        # ax_2.set_yscale("function", functions=(lambda x: np.log(x - 1), lambda x: np.exp(x) + 1))
        # formatter = ticker.FuncFormatter(lambda x, _: f"$1 + 10^{{{np.log10(x - 1):.0f}}}$")
        # ax_2.yaxis.set_major_formatter(formatter)
        # ax_2.yaxis.set_major_locator(OffsetLogLocator(1))
        ax_2.set_yscale("log")
        ax_2.set_ylim(1, 28)
        yticks = [1, 2, 5, 10, 25]
        ax_2.set_yticks(yticks)
        ax_2.set_yticklabels([str(yt) for yt in yticks])
        ax_2.set_ylabel(r"$\mu$", color="tab:red", rotation=0, labelpad=8)
        ax_2.tick_params(axis='y', labelcolor="tab:red")

        fig.tight_layout()

        plot_directory = Path(__file__).parent / "plot"
        plot_directory.mkdir(exist_ok=True)
        plot_path = plot_directory / f"subsample_statistics_{space}_{basis_name}.png"
        print("Saving subsample statistics plot to", plot_path)
        plt.savefig(
            plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
        )
        plt.close(fig)
