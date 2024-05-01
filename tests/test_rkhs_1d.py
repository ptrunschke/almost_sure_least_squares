import pytest
import numpy as np
from numpy.polynomial.legendre import legval

from rkhs_1d import h1_kernel, kernel_matrix


@pytest.fixture(params=[-0.75, -0.1, 0.3, 1.0])
def test_point(request):
    return request.param


@pytest.fixture(params=[10])
def test_dimension(request):
    return request.param


def test_kernel_matrix():
    xs = np.linspace(-1, 1, 1000)
    K = kernel_matrix(h1_kernel, xs)
    assert K.shape == (len(xs), len(xs))
    assert np.allclose(K, K.T)
    assert np.all(np.linalg.eigvalsh(K) >= 0)


def test_kernel(test_point, test_dimension):
    xs = np.linspace(-1, 1, 1000)
    ks = h1_kernel(xs, test_point)
    assert ks.shape == (len(xs),)
    measures = legval(xs, np.eye(test_dimension))
    assert measures.shape == (test_dimension, len(xs))

    l2_inner = 0.5 * np.trapz(measures * ks, xs, axis=1)
    assert l2_inner.shape == (test_dimension,)
    h = np.diff(xs)[0]
    assert np.allclose(np.diff(xs), h)
    d_ks = np.diff(ks) / h
    d_measures = np.diff(measures, axis=1) / h
    h10_inner = 0.5 * np.sum(d_measures * d_ks * h, axis=1)
    assert h10_inner.shape == (test_dimension,)

    inner = l2_inner + h10_inner
    tol = 1 / len(xs)
    assert np.allclose(inner, legval(test_point, np.eye(test_dimension)), atol=tol)