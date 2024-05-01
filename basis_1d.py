import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import ufl
from dolfinx import fem, mesh
from ufl import dx, grad, inner

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
