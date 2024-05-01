# coding: utf-8
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
import scipy.sparse as sps
from scipy.special import erf
import matplotlib as mpl
import matplotlib.pyplot as plt

import ufl
from dolfinx import fem, mesh
from ufl import dx, grad, inner


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

def h1gauss_kernel(x, y):
    low = np.minimum(x, y)
    high = np.maximum(x, y)
    sq_norm = x**2 + y**2
    return np.sqrt(np.pi / 2) * np.exp(sq_norm / 2) * (1 + erf(low / np.sqrt(2))) * (1 - erf(high / np.sqrt(2)))

def compute_inner(space, domain, gridpoints):
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
        raise ValueError(f"invalid space for orthonormalise(): '{space}'")

    if space == "h10":
        I = inner(grad(u), grad(v)) * rho * dx
    elif space in ["l2", "l2gauss"]:
        I = inner(u, v) * rho * dx
    elif space in ["h1", "h1gauss"]:
        I = inner(grad(u), grad(v)) * rho * dx
        I += inner(u, v) * rho * dx
    else:
        raise ValueError(f"invalid space for orthonormalise(): '{space}'")
    I = fem.assemble_matrix(fem.form(I), bcs=bcs).to_scipy()

    xs = fe_domain.geometry.x
    assert xs.shape == (gridpoints, 3)
    assert np.all(xs[:, 1:] == 0)
    xs = xs[:, 0]
    assert np.all(xs[:-1] <= xs[1:])
    return I, xs

def orthonormalise(basisval, space, domain, gridpoints):
    I, xs = compute_inner(space, domain, gridpoints)

    basis = basisval(xs)
    assert basis.ndim == 2 and basis.shape[1] == gridpoints
    assert I.shape == (gridpoints, gridpoints)
    dimension = basis.shape[0]
    gramian = basis @ I @ basis.T
    assert gramian.shape == (dimension, dimension)
    es, vs = np.linalg.eigh(gramian)
    mask = es > 1e-12
    vs = vs[:, mask] / np.sqrt(es[mask])

    def orthogonal_basisval(x):
        return vs.T @ basisval(x)

    return orthogonal_basisval

if __name__ == "__main__":
    # space = "h1"
    # space = "h10"
    space = "h1gauss"

    if space == "h10":
        domain = (-1, 1)
        reference_kernel = h10_kernel
    elif space == "h1":
        domain = (-1, 1)
        reference_kernel = h1_kernel
    elif space == "h1gauss":
        domain = (-5, 5)
        reference_kernel = h1gauss_kernel
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
    # if space == "h1gauss":
    #     plt.yscale("log")
    # plt.show()

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

    dimension = 5
    # basis = "polynomial"
    basis = "fourier"

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

    if space == "h10":
        if basis == "polynomial":
            def basisval(x):
                values = monomval(x, np.eye(dimension))
                return values * (x - 1) * (x + 1)
        elif basis == "fourier":
            def basisval(x):
                x = (x - domain[0]) / (domain[1] - domain[0])
                return np.sin(np.pi * x[None] * np.arange(1, dimension + 1)[:, None])
        else:
            raise NotImplementedError()
    else:
        if basis == "polynomial":
            def basisval(x):
                return monomval(x, np.eye(dimension))
        elif basis == "fourier":
            def basisval(x):
                x = (x - domain[0]) / (domain[1] - domain[0])
                c = dimension // 2 + (dimension % 2)
                s = dimension // 2
                assert c + s == dimension
                z = np.ones((1, len(x)))
                c = np.cos(2 * np.pi * x[None] * np.arange(1, c)[:, None])
                s = np.sin(2 * np.pi * x[None] * np.arange(1, s + 1)[:, None])
                assert z.shape[0] + c.shape[0] + s.shape[0] == dimension
                return np.concatenate([z, c, s], axis=0)
        else:
            raise NotImplementedError()

    # xs = np.linspace(*domain, 1000)
    # for e, b in enumerate(basisval(xs)):
    #     plt.plot(xs, b, label=f"{e}")
    # plt.legend()
    # plt.show()

    if space == "h1gauss":
        l2_basis = orthonormalise(basisval, "l2gauss", domain, 2 ** 12)
    else:
        l2_basis = orthonormalise(basisval, "l2", domain, 2 ** 12)

    rkhs_basis = orthonormalise(basisval, space, domain, 2 ** 12)

    xs = np.linspace(*domain, 1002)[1:-1]  # remove the boundary points for the H10 case
    M_onb = l2_basis(xs)
    I_onb = rkhs_basis(xs)

    # for e, b in enumerate(M_onb):
    #     plt.plot(xs, b, label=f"{e}")
    # plt.legend()
    # plt.show()

    ch = np.sum(M_onb ** 2, axis=0)
    if space == "h10":
        ch *= 0.5
    elif space == "h1":
        ch *= 0.5
    elif space == "h1gauss":
        def rho(x):
            return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

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
        plt.title(r"Polynomial basis for $H^1_0(-1, 1)$")
    elif space == "h1":
        plt.ylim(0.25, 0.8)
        plt.title(r"Polynomial basis for $H^1(-1, 1)$")
    elif space == "h1gauss":
        plt.ylim(-0.025, 0.25)
        plt.title(r"Polynomial basis for $H^1_w(\mathbb{R})$ with $w(x) = \frac{1}{\sqrt{2\pi}}\exp(-\frac{x^2}{2})$")
    plt.show()
