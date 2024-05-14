from abc import ABC, abstractmethod

import numpy as np
from scipy.special import erf
import matplotlib as mpl
import matplotlib.pyplot as plt


class Kernel(ABC):
    @property
    @abstractmethod
    def domain(self) -> tuple[float, float]:
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass


class H1Kernel(Kernel):
    def __init__(self, domain: tuple[float, float] = (-1, 1)):
        self._domain = domain

    @property
    def domain(self) -> tuple[float, float]:
        return self._domain

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        domain = self.domain
        assert np.all((x >= domain[0]) & (x <= domain[1]))
        assert np.all((y >= domain[0]) & (y <= domain[1]))
        low = np.minimum(x, y)
        high = np.maximum(x, y)
        return np.cosh(low - domain[0]) * np.cosh(domain[1] - high) * (domain[1] - domain[0]) / np.sinh(domain[1] - domain[0])


class H10Kernel(Kernel):
    def __init__(self, domain: tuple[float, float] = (-1, 1)):
        self._domain = domain

    @property
    def domain(self) -> tuple[float, float]:
        return self._domain

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        domain = self.domain
        assert np.all((x >= domain[0]) & (x <= domain[1]))
        assert np.all((y >= domain[0]) & (y <= domain[1]))
        d = domain[1] - domain[0]
        x = (x - domain[0]) / d
        y = (y - domain[0]) / d
        return d ** 2 * np.where(x <= y, x * (1 - y), (1 - x) * y)


class H1GaussKernel(Kernel):
    def __init__(self, domain: tuple[float, float] = (-np.inf, np.inf)):
        self._domain = domain

    @property
    def domain(self) -> tuple[float, float]:
        return self._domain

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert np.all((x >= self.domain[0]) & (x <= self.domain[1]))
        assert np.all((y >= self.domain[0]) & (y <= self.domain[1]))
        low = np.minimum(x, y)
        high = np.maximum(x, y)
        sq_norm = x**2 + y**2
        return np.sqrt(np.pi / 2) * np.exp(sq_norm / 2) * (1 + erf(low / np.sqrt(2))) * (1 - erf(high / np.sqrt(2)))


def kernel_matrix(kernel: Kernel, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    K = kernel(points[:, None], points[None, :])
    assert K.shape == (len(points), len(points))
    return K


if __name__ == "__main__":
    from pathlib import Path
    from basis_1d import compute_discrete_gramian, orthonormalise, enforce_zero_trace, MonomialBasis, FourierBasis, SinBasis

    # dimension = 5
    dimension = 10
    basis_name = "polynomial"
    # basis_name = "fourier"

    for space in ["h10", "h1", "h1gauss"]:
        print(f"Computing marginal densities for {space} with {basis_name} basis")
        if space == "h10":
            reference_kernel = H10Kernel((-1, 1))
            if basis_name == "polynomial":
                initial_basis = MonomialBasis(dimension, domain=(-1, 1))
            elif basis_name == "fourier":
                initial_basis = SinBasis(dimension, domain=(-1, 1))
            initial_basis = enforce_zero_trace(initial_basis)
        elif space == "h1":
            reference_kernel = H1Kernel((-1, 1))
            if basis_name == "polynomial":
                initial_basis = MonomialBasis(dimension, domain=(-1, 1))
            elif basis_name == "fourier":
                initial_basis = FourierBasis(dimension, domain=(-1, 1))
        elif space == "h1gauss":
            # reference_kernel = H1GaussKernel((-5, 5))
            reference_kernel = H1GaussKernel((-8, 8))
            if basis_name == "polynomial":
                # basisval = MonomialBasis(dimension, domain=(-5, 5))
                initial_basis = MonomialBasis(dimension, domain=(-8, 8))
            elif basis_name == "fourier":
                # basisval = FourierBasis(dimension, domain=(-5, 5))
                initial_basis = FourierBasis(dimension, domain=(-8, 8))
        else:
            raise NotImplementedError()

        # tab20 = mpl.colormaps["tab20"].colors
        # offset = 3
        # js = np.arange(nx)[offset:-offset].reshape(5, -1)[:, 0]
        # js = np.concatenate([js, [nx - offset]])
        # for e, j in enumerate(js):
        #     c = xs[j]
        #     ks = K[:, j]
        #     ks_ref = reference_kernel(xs, c)
        #     plt.plot(xs, L[:, j], color=tab20[2 * e])
        #     plt.plot(xs, ks, color=tab20[2 * e])
        #     plt.plot(xs, ks_ref, color=tab20[2 * e + 1], linestyle=":")
        # if space == "h1gauss":
        #     plt.yscale("log")
        # plt.show()

        # tab20 = mpl.colormaps["tab20"].colors
        # I, xs = compute_inner(space, domain, 2 ** 12)
        # L = sps.eye(I.shape[0], format="csc", dtype=float)
        # I = I.tocsc()
        # K = sps.linalg.spsolve(I, L).toarray()
        # L = L.toarray()
        # us = np.diag(K).copy()
        # us_ref = reference_kernel(xs, xs)
        # if space == "h10":
        #     us[[0, -1]] = 0
        # plt.plot(xs, us, color=tab20[0])
        # plt.plot(xs, us_ref, color=tab20[1], linestyle=":")
        # if space == "h1gauss":
        #     plt.yscale("log")
        # plt.show()

        if space == "h1gauss":
            discrete_l2_gramian = compute_discrete_gramian("l2gauss", initial_basis.domain, 2 ** 13)
        else:
            discrete_l2_gramian = compute_discrete_gramian("l2", initial_basis.domain, 2 ** 13)
        l2_basis = orthonormalise(initial_basis, *discrete_l2_gramian)

        discrete_rkhs_gramian = compute_discrete_gramian(space, initial_basis.domain, 2 ** 13)
        rkhs_basis = orthonormalise(initial_basis, *discrete_rkhs_gramian)

        # xs = np.linspace(*domain, 1000)
        # for e, b in enumerate(rkhs_basis(xs)):
        #     plt.plot(xs, b, label=f"{e}")
        # plt.legend()
        # plt.show()

        xs = np.linspace(*initial_basis.domain, 1002)[1:-1]  # remove the boundary points for the H10 case
        M_onb = l2_basis(xs)
        I_onb = rkhs_basis(xs)

        # for e, b in enumerate(M_onb):
        #     plt.plot(xs, b, label=f"{e}")
        # plt.legend()
        # plt.show()

        if space in ["h10", "h1"]:
            def rho(x):
                return np.full(len(x), 0.5)

        elif space == "h1gauss":
            def rho(x):
                return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

        normalise = lambda ys: ys / np.trapz(ys, xs)

        ch = np.sum(M_onb ** 2, axis=0)
        kd = np.sum(I_onb ** 2, axis=0)
        k = reference_kernel(xs, xs)
        ratio = np.nan_to_num(kd / k)

        plt.style.use('seaborn-v0_8-deep')
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
        ax.plot(xs, normalise(ratio), label=r"$\frac{k_d(x,x)}{k(x,x)}$")
        ax.plot(xs, normalise(k * rho(xs)), label=r"$k(x,x) \rho(x)$")
        ax.plot(xs, normalise(kd * rho(xs)), label=r"$k_d(x,x) \rho(x)$")
        ax.plot(xs, normalise(ch * rho(xs)), label="Christoffel density")
        # plt.plot(xs, normalise(ratio * rho(xs)), color="tab:purple", label=r"$\frac{k_d(x,x)}{k(x,x)} \rho$")
        ax.legend()
        ax.set_xlim(*initial_basis.domain)
        if space == "h10":
            # plt.title(r"Polynomial basis for $H^1_0(-1, 1)$")
            pass
        elif space == "h1":
            ax.set_ylim(0.25, 0.8)
            # plt.title(r"Polynomial basis for $H^1(-1, 1)$")
        elif space == "h1gauss":
            ax.set_ylim(-0.025, 0.25)
            # plt.title(r"Polynomial basis for $H^1_w(\mathbb{R})$ with $w(x) = \frac{1}{\sqrt{2\pi}}\exp(-\frac{x^2}{2})$")

        plot_directory = Path(__file__).parent / "plot"
        plot_directory.mkdir(exist_ok=True)
        plot_path = plot_directory / f"marginal_{space}_{basis_name}.png"
        print("Saving sample statistics plot to", plot_path)
        plt.savefig(
            plot_path, dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
        )
        plt.close(fig)
