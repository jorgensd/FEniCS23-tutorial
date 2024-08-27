# # Approximation of a finite element function
# We will now combine the previously presented components into a single script.
# To solve this problem, we create the variational form
# \begin{align}\int_\Omega uv~\mathrm{d}x = \int_\Omega \frac{f}{g}v~\mathrm{d}x\quad \forall v \in V\end{align}.

# ## Reminder: DOLFINx syntax
# As seen in previous classes and tutorials, you can solve this problem in DOLFINx for a 3rd order discontinuous Lagrange element with the following code

from mpi4py import MPI

import numpy as np
import pyvista

import dolfinx.fem.petsc
import ufl

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 3))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = u * v * ufl.dx

F = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 2))
f = dolfinx.fem.Function(F)
G = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
g = dolfinx.fem.Function(G)

L = f / g * v * ufl.dx

f.interpolate(lambda x: x[0])
g.interpolate(lambda x: 2 + np.sin(x[1]))

problem = dolfinx.fem.petsc.LinearProblem(a, L)
uh = problem.solve()


def plot_scalar_function(u: dolfinx.fem.Function):
    pyvista.start_xvfb(1.0)
    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))
    u_grid.point_data["u"] = u.x.array
    plotter = pyvista.Plotter()
    plotter.add_mesh(u_grid)
    plotter.show_axes()
    plotter.view_xy()
    plotter.show()


plot_scalar_function(uh)
