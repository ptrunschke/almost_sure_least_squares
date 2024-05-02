import pytest

import numpy as np
from scipy.linalg import solve_triangular
from numpy.polynomial.legendre import Legendre, legval
from joblib import Parallel, delayed


def evaluate(*args, **kwargs):
    def evaluate_function(f):
        return f(*args, **kwargs)
    return evaluate_function


def hk_gramian(dimension, k):
    def l2_inner(p1, p2):
        l2 = (p1*p2).integ()
        return (l2(1) - l2(-1)) / 2

    def entry(i, j):
        ret = 0
        bi, bj = Legendre.basis(i), Legendre.basis(j)
        for _ in range(k+1):
            ret += l2_inner(bi, bj)
            bi, bj = bi.deriv(), bj.deriv()
        return ret

    @Parallel(n_jobs=15)
    @evaluate()
    def all_entries():
        for i in range(dimension):
            for j in range(i+1):
                yield delayed(entry)(i, j)

    G = np.zeros((dimension, dimension))
    e = 0
    for i in range(dimension):
        for j in range(i+1):
            G[j,i] = G[i,j] = all_entries[e]
            e += 1

    return G


def orthonormal_basis(gramian):
    assert gramian.ndim == 2
    dimension = gramian.shape[0]
    assert gramian.shape[1] == dimension
    L = np.linalg.cholesky(gramian)
    assert np.allclose(L @ L.T, gramian)

    def evaluate_basis(points, coefficients):
        points = np.asarray(points)
        coefficients = np.asarray(coefficients)
        # assert points.ndim == 1
        assert coefficients.ndim <= 2
        assert coefficients.shape[0] <= dimension
        local_L = L[:coefficients.shape[0], :coefficients.shape[0]]
        coefficients = solve_triangular(local_L.T, coefficients, lower=True)
        return legval(points, coefficients)

    return evaluate_basis


@pytest.fixture(params=[3, 5, 10, 14, 30])
def test_gramian_dimension(request):
    return request.param


@pytest.fixture(params=[3, 5, 8])
def test_coefficient_dimension(request):
    return request.param


def test_l2_gramian(test_gramian_dimension):
    dimension = test_gramian_dimension
    gramian = hk_gramian(dimension, 0)
    assert gramian.shape == (dimension, dimension)
    factors = 1 / (2 * np.arange(dimension) + 1)
    assert np.allclose(gramian, np.diag(factors))

    xs = np.linspace(-1, 1, 1000)
    measures = legval(xs, np.eye(dimension))
    assert measures.shape == (dimension, len(xs))
    emp_gramian = 0.5 * np.trapz(measures[:, None, :] * measures[None, :, :], xs, axis=-1)
    assert emp_gramian.shape == (dimension, dimension)
    assert np.allclose(emp_gramian, np.diag(factors), atol=1 / len(xs))


def test_l2_legendre(test_gramian_dimension, test_coefficient_dimension):
    dimension = test_gramian_dimension
    local_dimension = test_coefficient_dimension
    if local_dimension > dimension:
        pytest.skip()
    gramian = hk_gramian(dimension, 0)
    l2_legendre = orthonormal_basis(gramian)
    xs = np.linspace(-1, 1, 1000)
    measures = l2_legendre(xs, np.eye(local_dimension))
    assert measures.shape == (local_dimension, len(xs))
    emp_gramian = 0.5 * np.trapz(measures[:, None, :] * measures[None, :, :], xs, axis=-1)
    assert emp_gramian.shape == (local_dimension, local_dimension)
    assert np.allclose(emp_gramian, np.eye(local_dimension), atol=1 / len(xs))


def derivative_operator(dimension):
    basis = Legendre.basis
    operator = np.zeros((dimension, dimension))
    for k in range(dimension):
        c = basis(k).deriv().coef
        operator[:len(c), k] = c
    return operator


def test_h1_gramian(test_gramian_dimension):
    dimension = test_gramian_dimension
    if dimension > 14:
        pytest.skip()
    l2_gramian = hk_gramian(dimension, 0)
    h1_gramian = hk_gramian(dimension, 1)
    assert l2_gramian.shape == (dimension, dimension)
    assert h1_gramian.shape == (dimension, dimension)
    D = derivative_operator(dimension)
    reference_gramian = l2_gramian + D.T @ l2_gramian @ D
    assert np.allclose(h1_gramian, reference_gramian)

    xs = np.linspace(-1, 1, 10_000)
    measures = legval(xs, np.eye(dimension))
    assert measures.shape == (dimension, len(xs))
    emp_gramian_l2 = 0.5 * np.trapz(measures[:, None, :] * measures[None, :, :], xs, axis=-1)
    assert emp_gramian_l2.shape == (dimension, dimension)
    assert np.allclose(emp_gramian_l2, np.diag(1 / (2 * np.arange(dimension) + 1)), atol=1 / len(xs), rtol=1 / len(xs))

    h = np.diff(xs)[0]
    assert np.allclose(np.diff(xs), h)
    d_measures = np.diff(measures, axis=1) / h
    emp_gramian_h10 = 0.5 * np.sum(d_measures[:, None, :] * d_measures[None, :, :] * h, axis=-1)
    assert emp_gramian_h10.shape == (dimension, dimension)

    emp_gramian_h10_ref = D.T @ emp_gramian_l2 @ D
    assert np.allclose(emp_gramian_h10, emp_gramian_h10_ref, atol=1 / len(xs), rtol=1 / len(xs))

    emp_gramian = emp_gramian_l2 + emp_gramian_h10
    assert np.allclose(emp_gramian, h1_gramian, atol=1 / len(xs))


def test_differential_operator(test_gramian_dimension):
    dimension = test_gramian_dimension
    if dimension > 14:
        pytest.skip()
    D = derivative_operator(dimension)
    assert D.shape == (dimension, dimension)
    xs = np.linspace(-1, 1, 10_000)
    measures = legval(xs, np.eye(dimension))
    assert measures.shape == (dimension, len(xs))
    h = np.diff(xs)[0]
    assert np.allclose(np.diff(xs), h)
    d_measures = np.diff(measures, axis=1) / h
    xs_mid = np.linspace(-1 + h / 2, 1 - h / 2, len(xs) - 1)
    measures_mid = legval(xs_mid, np.eye(dimension))
    assert measures_mid.shape == (dimension, len(xs_mid))
    assert np.allclose(d_measures, D.T @ measures_mid, atol=1 / len(xs), rtol=1 / len(xs))


def test_h1_legendre(test_gramian_dimension, test_coefficient_dimension):
    dimension = test_gramian_dimension
    local_dimension = test_coefficient_dimension
    if local_dimension > dimension or dimension > 14:
        pytest.skip()
    h1_gramian = hk_gramian(dimension, 1)
    h1_legendre = orthonormal_basis(h1_gramian)
    xs = np.linspace(-1, 1, 10_000)
    measures = h1_legendre(xs, np.eye(local_dimension))
    assert measures.shape == (local_dimension, len(xs))
    emp_gramian_l2 = 0.5 * np.trapz(measures[:, None, :] * measures[None, :, :], xs, axis=-1)
    assert emp_gramian_l2.shape == (local_dimension, local_dimension)

    h = np.diff(xs)[0]
    assert np.allclose(np.diff(xs), h)
    d_measures = np.diff(measures, axis=1) / h
    emp_gramian_h10 = 0.5 * np.sum(d_measures[:, None, :] * d_measures[None, :, :] * h, axis=-1)
    assert emp_gramian_h10.shape == (local_dimension, local_dimension)

    emp_gramian = emp_gramian_l2 + emp_gramian_h10
    # tol = 1 / len(xs) / np.linalg.cond(h1_gramian)
    # assert np.allclose(emp_gramian, np.eye(local_dimension), atol=tol, rtol=tol)
    assert np.max(abs(emp_gramian - np.eye(local_dimension))) < 1e-1 * np.linalg.cond(h1_gramian)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    dimension = 10

    l2_legendre = orthonormal_basis(hk_gramian(dimension, 0))
    h1_legendre = orthonormal_basis(hk_gramian(dimension, 1))

    xs = np.linspace(-1, 1, 1000)
    l2_christoffel = l2_legendre(xs, np.eye(dimension))
    l2_christoffel = (l2_christoffel**2).sum(axis=0)
    assert np.isclose(0.5 * np.trapz(l2_christoffel, xs), dimension, rtol=1 / len(xs))
    l2_christoffel /= dimension

    h1_christoffel = h1_legendre(xs, np.eye(dimension))
    h1_christoffel = (h1_christoffel**2).sum(axis=0)
    h1_christoffel /= 0.5 * np.trapz(h1_christoffel, xs)

    plt.plot(xs, l2_christoffel, label="L2")
    plt.plot(xs, h1_christoffel, label="H1")
    plt.show()
