# coding: utf-8
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
from dolfinx.io import gmshio

import pyvista

import gmsh
gmsh.initialize()

dimension = 2

# gmsh.model.add("L_shape")
# gmsh.model.occ.add_point(0, 0, 0, tag=1)
# gmsh.model.occ.add_point(2, 0, 0, tag=2)
# gmsh.model.occ.add_point(2, 2-1e-5, 0, tag=3)
# gmsh.model.occ.add_point(1, 1, 0, tag=4)
# gmsh.model.occ.add_point(2-1e-5, 2, 0, tag=5)
# gmsh.model.occ.add_point(0, 2, 0, tag=6)
# gmsh.model.occ.add_line(1, 2, tag=1)
# gmsh.model.occ.add_line(2, 3, tag=2)
# gmsh.model.occ.add_line(3, 4, tag=3)
# gmsh.model.occ.add_line(4, 5, tag=4)
# gmsh.model.occ.add_line(5, 6, tag=5)
# gmsh.model.occ.add_line(6, 1, tag=6)
# gmsh.model.occ.add_curve_loop([1, 2, 3, 4, 5, 6], tag=1)
# gmsh.model.occ.add_plane_surface([1], tag=1)
# gmsh.model.occ.synchronize()
# gmsh.model.add_physical_group(0, [1, 2, 3, 4, 5, 6], tag=1, name="points")
# gmsh.model.add_physical_group(1, [1, 2, 3, 4, 5, 6], tag=2, name="lines")
# gmsh.model.add_physical_group(2, [1], tag=3, name="facets")
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
# gmsh.model.mesh.generate(dimension)
# gmsh_model_rank = 0
# mesh_comm = MPI.COMM_WORLD
# domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=dimension)
# gmsh.finalize()

gmsh.model.occ.add_rectangle(0, 0, 0, 2, 1, tag=1)
gmsh.model.occ.add_rectangle(0, 0, 0, 1, 2, tag=2)
gmsh.model.occ.fuse([(dimension, 1)], [(dimension, 2)], tag=3)
gmsh.model.occ.synchronize()
gmsh.model.add_physical_group(dimension, [3], tag=1, name="L_shape")
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(dimension)
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=dimension)
gmsh.finalize()

V = fem.functionspace(domain, ("Lagrange", 1))

# facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
#                                        marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
#                                                                       np.isclose(x[0], 2.0)))
# dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
# bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
S = inner(grad(u), grad(v)) * dx
M = inner(u, v) * dx

S_mat = fem.assemble_matrix(fem.form(S)).to_scipy()
M_mat = fem.assemble_matrix(fem.form(M)).to_scipy()

# The linear form should just be the evaluation of the function f at the point x.
# Therefore, it is just a standard basis vector.

import scipy.sparse as sps
L_mat = sps.eye(S_mat.shape[0], format="csc", dtype=float)
X = sps.linalg.spsolve(S_mat + M_mat, L_mat)
X = X.todense()
uh = fem.Function(V)
uh.vector.array[:] = np.diag(X)

# a = S + M
# L = inner(f, v) * dx # / (inner(f, f) * dx)

# # problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# problem = LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# uh = problem.solve()

cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
# grid.point_data["u"] = f.vector.array.real
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_scalar()
plotter.add_mesh(warped)
plotter.show()
