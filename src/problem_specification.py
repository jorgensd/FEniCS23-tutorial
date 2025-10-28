# # Approximation of a finite element function in {term}`DOLFINx`
#
# In this section, we will combine all the components presented in the previous sections to solve the projection problem
# To solve this problem, we create the variational form
#
# $$
# \int_\Omega uv~\mathrm{d}x = \int_\Omega \frac{f}{g}v~\mathrm{d}x\quad \forall v \in V.
# $$

# # Creating a simple mesh in {term}`DOLFINx`
# As you might have seen in other FEniCSx tutorials, there are some "built-in" meshes in DOLFINx.
# In this tutorial, we will use a unit square, consisting of triangular elements,
# where we have 10 elements in each direction.
# To create a mesh, we need to decide on what {term}`MPI` communicator we want to use.
# For all the examples in this tutorial, we will use the communicator
# {py:obj}`MPI.COMM_WORLD<mpi4py.MPI.COMM_WORLD>`.

from mpi4py import MPI

import numpy as np
import pyvista

import dolfinx.fem.petsc
import ufl

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

# Next, we create the function space for our unknown `u`. In this example,
# we use a discontinuous Lagrange element of degree 3.

V = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 3))

# Next, we can write the bi-linear form `a` and the linear form `L`.

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = u * v * ufl.dx

# ```{note}
# Compared to the previous sections, we have now used
# {py:class}`dolfinx.fem.Function`, instead of
# {py:class}`ufl.Coefficient` to defined `f` and `g`.
# This is because a {py:class}`ufl.Coefficient` does not hold any data.
# The {py:class}`dolfinx.fem.Function` is sub-classed from
# {py:class}`ufl.Coefficient`
# and holds data for the degrees of freedom.
# ````

F = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 2))
f = dolfinx.fem.Function(F)
G = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
g = dolfinx.fem.Function(G)

L = f / g * v * ufl.dx

# We populate the degrees of freedom of `f` and `g` with some data.
# We use a [lambda-function](https://discuss.python.org/t/what-is-the-purpose-of-lambda-expressions/12415)
# to define an operator that takes in a set of points `x` as a `numpy` 2D array
# (of the shape `(3, num_points)`) and returns an array of shape `num_points` with the data at the input points.
#
# ```{note}
# In DOLFINx, the input data to the interpolation operator has the shape `(3, num_points)` and not `(num_points, 3)`.
# This might seem confusing at first, as you can't access the `i`th point as `x[i]`,
# but you have to access it as `x[:, i]`.
# However, this is beneficial when you can use vectorized operations, such as `numpy.sin`,
# which can operate on the entire array at once.
# ```

g.interpolate(lambda x: 0.8 + np.sin(np.pi * x[1]))

# As `f` is in a discontinuous function space, we will create a piecewise constant function over parts of the domain.
# We do this by first locating what cells satisfies `x<=0.5` (i.e. all vertices of the cell satisfies this condition).

left_cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, lambda x: x[0] <= 0.5 + 1e-14)

# Next we populate all degrees of freedom by the function we want in the rest of the domain, $f(x,y) = x$.

f.interpolate(lambda x: x[0])

# And then we overwrite the values in the left cells with the function $f(x,y) = 1$.

f.interpolate(lambda x: np.full(x.shape[1], 1, dtype=dolfinx.default_scalar_type), cells0=left_cells)

# As we are only working on a sub-set of cells,
# we call {py:meth}`scatter_forward<dolfinx.la.Vector.scatter_forward>`
# to ensure that all processes on a distributed system gets the correct values.

f.x.scatter_forward()

# Next we need to solve the linear problem. We do this using {term}`PETSc`,
# which is an interface to many linear algebra libraries,
# as well as providing it's own set of solvers.
# In DOLFINx, we provide a simple interface to {py:mod}`PETSc<petsc4py.PETSc>`
# with the {py:class}`dolfinx.fem.petsc.LinearProblem` class.
# If we want to give PETSc some options, such as what direct solver to use, we can do this with a dictionary.
# In the following example, we will use a direct solver, and specifically the {term}`MUMPS` solver.

petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
problem = dolfinx.fem.petsc.LinearProblem(a, L, petsc_options=petsc_options, petsc_options_prefix="projection_")

# Now we can solve the problem and plot the solution.

uh = problem.solve()


def plot_scalar_function(u: dolfinx.fem.Function):
    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))
    u_grid.point_data["u"] = u.x.array
    linear_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space.mesh))
    warped = u_grid.warp_by_scalar()
    plotter = pyvista.Plotter()
    plotter.add_mesh(linear_grid, style="wireframe", color="black")
    plotter.add_mesh(warped)
    plotter.show_axes()
    plotter.show()


plot_scalar_function(uh)
