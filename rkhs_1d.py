# coding: utf-8
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import scipy.sparse as sps
import matplotlib as mpl
import matplotlib.pyplot as plt

import ufl
from dolfinx import fem, mesh
from ufl import dx, grad, inner

# space = "h1"
# space = "h10"
space = "wh1"

nx = 2 ** 12
if space == "h10":
    domain = mesh.create_interval(comm=MPI.COMM_WORLD, points=(-1.0, 1.0), nx=nx-1)
elif space == "h1":
    domain = mesh.create_interval(comm=MPI.COMM_WORLD, points=(-1.0, 1.0), nx=nx-1)
elif space == "wh1":
    domain = mesh.create_interval(comm=MPI.COMM_WORLD, points=(-5.0, 5.0), nx=nx-1)
else:
    raise NotImplementedError()

V = fem.functionspace(domain, ("Lagrange", 1))

if space == "h10":
    facets = mesh.locate_entities_boundary(domain, dim=0,
                                           marker=lambda x: np.isclose(x[0], -1.0) | np.isclose(x[0],  1.0))
    dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
    bcs = [bc]
elif space == "h1":
    bcs = None
elif space == "wh1":
    bcs = None
else:
    raise NotImplementedError()

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

if space == "h10":
    S = inner(grad(u), grad(v)) * 0.5 * dx
    M = inner(u, v) * 0.5 * dx
    I = fem.assemble_matrix(fem.form(S), bcs=bcs).to_scipy()
    M = fem.assemble_matrix(fem.form(M), bcs=bcs).to_scipy()
elif space == "h1":
    S = inner(grad(u), grad(v)) * 0.5 * dx
    M = inner(u, v) * 0.5 * dx
    I = fem.assemble_matrix(fem.form(S + M), bcs=bcs).to_scipy()
    M = fem.assemble_matrix(fem.form(M), bcs=bcs).to_scipy()
elif space == "wh1":
    x = ufl.SpatialCoordinate(domain)
    rho = ufl.exp(-x[0]**2 / 2) / np.sqrt(2 * np.pi)
    S = inner(grad(u), grad(v)) * rho * dx
    M = inner(u, v) * rho * dx
    I = fem.assemble_matrix(fem.form(S + M), bcs=bcs).to_scipy()
    M = fem.assemble_matrix(fem.form(M), bcs=bcs).to_scipy()
else:
    raise NotImplementedError()
L = sps.eye(I.shape[0], format="csc", dtype=float)
I = I.tocsc()
K = sps.linalg.spsolve(I, L).toarray()
L = L.toarray()

xs = domain.geometry.x
assert xs.shape == (nx, 3)
assert np.all(xs[:, 1:] == 0)
xs = xs[:, 0]
assert np.all(xs[:-1] <= xs[1:])

def h1_kernel(x, y):
    domain = (-1, 1)
    assert np.all((x >= domain[0]) & (x <= domain[1]))
    assert np.all((y >= domain[0]) & (y <= domain[1]))
    low = np.minimum(x, y)
    high = np.maximum(x, y)
    return np.cosh(low - domain[0]) * np.cosh(domain[1] - high) * (domain[1] - domain[0]) / np.sinh(domain[1] - domain[0])

def h10_kernel(x, y):
    domain = (-1, 1)
    assert np.all((x >= domain[0]) & (x <= domain[1]))
    assert np.all((y >= domain[0]) & (y <= domain[1]))
    x = (x + 1) / 2
    y = (y + 1) / 2
    # 4 = 2 * 2, where one factor comes from the probability and the other comes from the transformation.
    return 4 * np.where(x <= y, x * (1 - y), (1 - x) * y)
    # If x <= y: (x + 1) / 2 * (1 - (y + 1) / 2) = (x + 1) * (1 - y) / 4
    # If x >  y: (1 - (x + 1) / 2) * (y + 1) / 2 = (1 - x) * (y + 1) / 4
    # --> (min(x, y) + 1) * (1 - max(x, y)) / 4

if space == "h10":
    reference_kernel = h10_kernel
elif space == "h1":
    reference_kernel = h1_kernel
elif space == "wh1":
    from scipy.special import erf

    # reference_kernel = lambda x, y: np.full_like(x * y, np.nan)

    # def reference_kernel(x, y):
    #     rho = lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    #     k_bare_left = lambda x: np.sqrt(np.pi / 2) * np.exp(x**2 / 2) * (erf(x / np.sqrt(2)) + 1)
    #     # c_left = lambda y: 1 / (rho(y) * (y * k_bare_left(y) + 1))
    #     k_bare_right = lambda x: np.sqrt(np.pi / 2) * np.exp(x**2 / 2) * (erf(x / np.sqrt(2)) - 1)
    #     # c_right = lambda y: -1 / (rho(y) * (y * k_bare_right(y) + 1))
    #     kyl = k_bare_left(y)
    #     kyr = k_bare_right(y)
    #     dkyl = rho(y) * (y * kyl + 1)
    #     dkyr = rho(y) * (y * kyr + 1)
    #     # system = np.array([[kyl, -kyr], [dkyl, -dkyr]])
    #     # cl, cr = np.linalg.solve(system, np.array([0, 1]))
    #     c_factor = (erf(y / np.sqrt(2)) + 1) / (erf(y / np.sqrt(2)) - 1)
    #     cl = 1 / (dkyl - c_factor * dkyr)
    #     cr = cl * c_factor
    #     return np.where(x <= y, cl * k_bare_left(x), cr * k_bare_right(x))

    def rho(x):
        return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    def reference_kernel(x, y):
        # k_left = lambda x: np.sqrt(np.pi / 2) * np.exp(x**2 / 2) * (erf(x / np.sqrt(2)) + 1) * (erf(y / np.sqrt(2)) - 1)
        # k_right = lambda x: np.sqrt(np.pi / 2) * np.exp(x**2 / 2) * (erf(x / np.sqrt(2)) - 1) * (erf(y / np.sqrt(2)) + 1)
        # assert np.allclose(k_left(y), k_right(y))
        # c = - 1 / (2 * rho(y))
        # return c * np.where(x <= y, k_left(x), k_right(x))
        k_sym = lambda x, y: np.sqrt(np.pi / 2) * np.exp((x**2 + y**2) / 2) * (1 + erf(x / np.sqrt(2))) * (1 - erf(y / np.sqrt(2)))
        # k_left = lambda x: k_sym(x, y)
        # k_right = lambda x: k_sym(y, x)
        # return np.where(x <= y, k_left(x), k_right(x))
        return k_sym(np.minimum(x, y), np.maximum(x, y))
else:
    raise NotImplementedError()


tab20 = mpl.colormaps["tab20"].colors

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
# if space == "wh1":
#     plt.yscale("log")
# plt.show()


# us = np.diag(K).copy()
# us_ref = reference_kernel(xs, xs)
# if space == "h10":
#     us[[0, -1]] = 0
# plt.plot(xs, us, color=tab20[0])
# plt.plot(xs, us_ref, color=tab20[1], linestyle=":")
# if space == "wh1":
#     plt.yscale("log")
# plt.show()


def basisval(x, c, tensor=True):
    assert tensor
    dimension, *c_shape = c.shape
    c = c.reshape(dimension, -1)
    x_shape = x.shape
    x = x.reshape(-1)
    measures = x[None] ** np.arange(dimension)[:, None]
    assert measures.shape == (dimension, x.size)
    values = c.T @ measures
    return values.reshape(*c_shape, *x_shape)

if space == "h10":
    _basisval = basisval

    def basisval(x, c, tensor=True):
        values = _basisval(x, c, tensor=tensor)
        return values * (x - 1) * (x + 1)

dimension = 5
basis = basisval(xs, np.eye(dimension))

def orthogonalise(basis, inner):
    dimension, discretisation = basis.shape
    assert inner.shape == (discretisation, discretisation)
    gramian = basis @ inner @ basis.T
    assert gramian.shape == (dimension, dimension)
    es, vs = np.linalg.eigh(gramian)
    vs = vs / np.sqrt(es)
    assert np.allclose(basisval(xs, vs), vs.T @ basis)
    return vs.T @ basis

I_onb = orthogonalise(basis, I)
M_onb = orthogonalise(basis, M)

# for e, b in enumerate(I_onb):
#     plt.plot(xs, b, label=f"{e}")
# plt.legend()
# plt.show()

ch = np.sum(M_onb ** 2, axis=0)
if space == "h10":
    ch *= 0.5
elif space == "h1":
    ch *= 0.5
elif space == "wh1":
    ch *= rho(xs)

normalise = lambda ys: ys / np.trapz(ys, xs)

kd = np.sum(I_onb ** 2, axis=0)
k = reference_kernel(xs, xs)
ratio = np.nan_to_num(kd / k)
plt.plot(xs, normalise(kd), color=tab20[0], label="$k_d(x,x)$")
plt.plot(xs, normalise(k), color=tab20[1], label="$k(x,x)$")
plt.plot(xs, normalise(ratio), color="tab:red", label=r"$\frac{k_d(x,x)}{k(x,x)}$")
# plt.plot(xs, normalise(ratio * rho(xs)), color="tab:purple", label=r"$\frac{k_d(x,x)}{k(x,x)} \rho$")
plt.plot(xs, normalise(ch), color="k", linestyle="--", label="Christoffel density")
plt.legend()
if space == "h10":
    plt.title("Polynomial basis for $H^1_0(-1, 1)$")
elif space == "h1":
    plt.ylim(0.25, 0.8)
    plt.title("Polynomial basis for $H^1(-1, 1)$")
elif space == "wh1":
    plt.ylim(-0.025, 0.25)
    plt.title(r"Polynomial basis for $H^1_w(\mathbb{R})$ with $w(x) = \frac{1}{\sqrt{2\pi}}\exp(-\frac{x^2}{2})$")
plt.show()
