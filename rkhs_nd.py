import numpy as np

from rkhs_1d import Kernel


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
