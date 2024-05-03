import numpy as np
from basis_1d import Basis
from rkhs_1d import Kernel, kernel_matrix

def optimal_least_squares(points: np.ndarray, values: np.ndarray, kernel: Kernel, basis: Basis) -> np.ndarray:
    points = np.asarray(points)
    values = np.asarray(values)
    assert points.ndim == 1 and points.shape == values.shape
    M = basis(points)
    assert M.shape == (basis.dimension, len(points))
    K = kernel_matrix(kernel, points)
    es, vs = np.linalg.eigh(K)
    assert np.allclose(vs * es @ vs.T, K)
    assert np.all(es >= 1e-8)
    K_plus = vs / es @ vs.T
    V = M @ K_plus @ M.T
    v = M @ K_plus @ values
    return np.linalg.solve(V, v)
