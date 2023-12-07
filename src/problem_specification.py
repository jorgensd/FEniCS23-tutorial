# # Problem statement
# The first problem we will encounter in this tutorial is an approximation problem.
# Given to function $f$ and $g$, compute the approximation of $\frac{f}{g}$ in a specified finite
# element space.
# Mathematically, we can write@
# Find $u\in V(\Omega)$ such that
# \begin{align}
# u &= \frac{f(x,y,z)}{g(x,y,z)} \qquad \text{in } \Omega\subset \mathbb{R}^3.
# \end{align}
#
# We will first show how we can easily solve this in FEniCSx on a unit square.
# Then, we will go through what happens under the hood in the various components listed on the [front page](../README).
# To solve this problem, we create the variational form
# \begin{align}\int_\Omega uv~\mathrm{d}x = \int_\Omega \frac{f}{g}v~\mathrm{d}x\quad \forall v \in V\end{align}.

# ## Reminder: DOLFINx syntax
# As seen in previous classes and tutorials, you can solve this problem in DOLFINx for a 3rd order discontinuous Lagrange element with the following code

import pyvista
import dolfinx.fem.petsc
import dolfinx
from mpi4py import MPI
import ufl
import numpy as np

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = dolfinx.fem.FunctionSpace(mesh, ("Discontinuous Lagrange", 3))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = u * v * ufl.dx

F = dolfinx.fem.FunctionSpace(mesh, ("Discontinuous Lagrange", 2))
f = dolfinx.fem.Function(F)
G = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 2))
g = dolfinx.fem.Function(G)

L = f / g * v * ufl.dx

f.interpolate(lambda x: x[0])
g.interpolate(lambda x: 2+np.sin(x[1]))

problem = dolfinx.fem.petsc.LinearProblem(a, L)
uh = problem.solve()


def plot_scalar_function(u: dolfinx.fem.Function):
    pyvista.start_xvfb(1.0)
    u_grid = pyvista.UnstructuredGrid(
        *dolfinx.plot.vtk_mesh(u.function_space))
    u_grid.point_data["u"] = u.x.array
    plotter = pyvista.Plotter()
    plotter.add_mesh(u_grid)
    plotter.show_axes()
    plotter.view_xy()
    plotter.show()


plot_scalar_function(uh)
