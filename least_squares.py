import numpy as np
from basis_1d import Basis
from rkhs_1d import Kernel, kernel_matrix

def optimal_least_squares(points: np.ndarray, values: np.ndarray, kernel: Kernel, basis: Basis) -> np.ndarray:
    if kernel.domain != basis.domain:
        raise ValueError(f"domain mismatch: kernel domain is {kernel.domain} but basis domain is {basis.domain}")
    dimension = np.reshape(kernel.domain, (-1, 2)).shape[0]
    points = np.asarray(points)
    values = np.asarray(values)
    assert points.ndim >= 1
    if points.ndim == 1 and dimension != 1:
        raise ValueError(f"dimension mismatch: input dimension is 1 but space dimension is {dimension}")
    elif points.shape[1] != dimension:
        raise ValueError(f"dimension mismatch: input dimension is {points.shape[1]} but space dimension is {dimension}")
    assert values.shape == (points.shape[0],)
    M = basis(points)
    assert M.shape == (basis.dimension, len(points))
    K = kernel_matrix(kernel, points)
    es, vs = np.linalg.eigh(K)
    assert np.allclose(vs * es @ vs.T, K)
    assert np.all(es >= 1e-8), [np.min(es), np.max(es)]
    K_plus = vs / es @ vs.T
    V = M @ K_plus @ M.T
    v = M @ K_plus @ values
    return np.linalg.solve(V, v)
