from abc import ABC, abstractmethod

import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import ufl
from dolfinx import fem, mesh
from ufl import dx, grad, inner


class Basis(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def domain(self) -> tuple[float, float]:
        pass

    @abstractmethod
    def __call__(self, points: np.ndarray) -> np.ndarray:
        pass

class TransformedBasis(Basis):
    def __init__(self, transform: np.ndarray, basis: Basis):
        assert transform.ndim == 2 and transform.shape[0] <= transform.shape[1] and transform.shape[1] == basis.dimension
        self.transform = transform
        self.basis = basis
        self._domain = basis.domain

    @property
    def dimension(self) -> int:
        return self.transform.shape[0]

    @property
    def domain(self) -> tuple[float, float]:
        return self._domain

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return np.tensordot(self.transform, self.basis(points), 1)


def monomval(x, c, tensor=True):
    assert tensor
    dimension, *c_shape = c.shape
    c = c.reshape(dimension, -1)
    x_shape = x.shape
    x = x.reshape(-1)
    measures = x[None] ** np.arange(dimension)[:, None]
    assert measures.shape == (dimension, x.size)
    values = c.T @ measures
    return values.reshape(*c_shape, *x_shape)


class MonomialBasis(Basis):
    def __init__(self, dimension: int, domain: tuple[float, float] = (-np.inf, np.inf)):
        self._dimension = dimension
        self._domain = domain

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def domain(self) -> tuple[float, float]:
        return self._domain

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return monomval(points, np.eye(self.dimension))


class SinBasis(Basis):
    def __init__(self, dimension: int, domain: tuple[float, float] = (-1, 1)):
        self._dimension = dimension
        self._domain = domain

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def domain(self) -> tuple[float, float]:
        return self._domain

    def __call__(self, points: np.ndarray) -> np.ndarray:
        x = points.reshape(-1)
        x = (x - self.domain[0]) / (self.domain[1] - self.domain[0])
        res = np.sin(np.pi * x[None] * np.arange(1, self.dimension + 1)[:, None])
        assert res.shape == (self.dimension, x.size)
        return res.reshape(self.dimension, *points.shape)


class FourierBasis(Basis):
    def __init__(self, dimension: int, domain: tuple[float, float] = (-1, 1)):
        self._dimension = dimension
        self._domain = domain

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def domain(self) -> tuple[float, float]:
        return self._domain

    def __call__(self, points: np.ndarray) -> np.ndarray:
        x = points.reshape(-1)
        x = (x - self.domain[0]) / (self.domain[1] - self.domain[0])
        c = self.dimension // 2 + (self.dimension % 2)
        s = self.dimension // 2
        assert c + s == self.dimension
        z = np.ones((1, x.size))
        c = np.cos(2 * np.pi * x[None] * np.arange(1, c)[:, None])
        s = np.sin(2 * np.pi * x[None] * np.arange(1, s + 1)[:, None])
        res = np.concatenate([z, c, s], axis=0)
        assert res.shape == (self.dimension, x.size)
        return res.reshape(self.dimension, *points.shape)


def compute_discrete_gramian(space: str, domain: tuple[float, float], gridpoints: int) -> tuple[np.ndarray, np.ndarray]:
    fe_domain = mesh.create_interval(comm=MPI.COMM_WORLD, points=domain, nx=gridpoints-1)
    V = fem.functionspace(fe_domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    if space == "h10":
        near_boundary = lambda x: np.isclose(x[0], domain[0]) | np.isclose(x[0],  domain[1])
        facets = mesh.locate_entities_boundary(fe_domain, dim=0, marker=near_boundary)
        dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=facets)
        bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
        bcs = [bc]
        rho = 1 / (domain[1] - domain[0])
    elif space in ["l2", "h1"]:
        bcs = None
        rho = 1 / (domain[1] - domain[0])
    elif space in ["l2gauss", "h1gauss"]:
        bcs = None
        x = ufl.SpatialCoordinate(fe_domain)
        rho = ufl.exp(-x[0]**2 / 2) / np.sqrt(2 * np.pi)
    else:
        raise ValueError(f"invalid space for compute_discrete_gramian(): '{space}'")

    if space == "h10":
        I = inner(grad(u), grad(v)) * rho * dx
    elif space in ["l2", "l2gauss"]:
        I = inner(u, v) * rho * dx
    elif space in ["h1", "h1gauss"]:
        I = inner(grad(u), grad(v)) * rho * dx
        I += inner(u, v) * rho * dx
    else:
        raise ValueError(f"invalid space for compute_discrete_gramian(): '{space}'")
    I = fem.assemble_matrix(fem.form(I), bcs=bcs).to_scipy()

    xs = fe_domain.geometry.x
    assert xs.shape == (gridpoints, 3)
    assert np.all(xs[:, 1:] == 0)
    xs = xs[:, 0]
    assert np.all(xs[:-1] <= xs[1:])
    return I, xs


def orthonormalise(basis: Basis, discrete_gramian: np.ndarray, discretisation: np.ndarray) -> Basis:
    assert discretisation.ndim == 1
    assert discrete_gramian.shape == (len(discretisation), len(discretisation))
    assert np.all(discretisation[:-1] < discretisation[1:])

    basisval = basis(discretisation)
    assert basisval.shape == (basis.dimension, len(discretisation))
    gramian = basisval @ discrete_gramian @ basisval.T
    es, vs = np.linalg.eigh(gramian)
    mask = es > 1e-12
    vs = vs[:, mask] / np.sqrt(es[mask])

    res = TransformedBasis(vs.T, basis)
    res._domain = discretisation[0], discretisation[-1]
    return res


def enforce_zero_trace(basis: Basis, tolerance: float = 1e-8) -> Basis:
    assert np.all(np.isfinite(basis.domain))
    trace = basis(np.array(basis.domain))
    assert trace.shape == (basis.dimension, 2)
    trace = trace.T
    U, s, Vt = np.linalg.svd(trace)
    assert np.allclose(U * s @ Vt[:2], trace)
    mask = np.full(basis.dimension, True)
    mask[:2] = s <= tolerance

    res = TransformedBasis(Vt[mask], basis)
    assert np.allclose(res(np.array(res.domain)), 0)
    return res


def create_subspace_kernel(basis: Basis):
    def subspace_kernel(x, y):
        x_measures = basis(x)
        dimension = x_measures.shape[0]
        assert x_measures.shape == (dimension,) + x.shape
        y_measures = basis(y)
        assert y_measures.shape == (dimension,) + y.shape
        return (x_measures * y_measures).sum(axis=0)

    return subspace_kernel
