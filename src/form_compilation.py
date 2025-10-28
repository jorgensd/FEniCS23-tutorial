# # Efficient usage of the Unified Form Language
#
# As we discussed in [Code generation](./code_generation)
# DOLFINx generates code once the user has specified the variational form.
# This process can be somewhat time consuming, as generating,
# compiling and linking the code can take time.
# We start by setting up a simple unit square and a first order Lagrange space.

# +
from mpi4py import MPI

import numpy as np

import dolfinx
import dolfinx.fem.petsc
import ufl

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 30, 30)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
# -

# ## Problem statement
# Let us consider a simple heat equation,
# \begin{align}
# \frac{\partial u}{\partial t} - \nabla \cdot( k(t) \nabla u) &= f(x,y,t) \qquad \text{in } \Omega \\
# \frac{\partial u}{\partial n}  &= 0 \qquad \text{on } \partial \Omega\\
# u(\cdot, 0) &= \frac{1}{2\pi \sigma} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2} e^{-\frac{1}{2}\left(\frac{y-\mu}{\sigma}\right)^2} \\
# k(t) &= \begin{cases}
# 0.1 \quad \text{if } t<0.5\\
# 0.05 \quad \text{if } t>=0.5
# \end{cases}\\
# f(x,y,t) &= \begin{cases}
# 0.4\cdot y \quad \text{if } x<0.5\\
# 0.5\cdot t\quad \text{if } x>=0.5
# \end{cases}
# \end{align}

# In this equation, there are several time dependent and spatially varying coefficients.
# With a naive implementation of this code in DOLFINx, one would define the variational formulation **inside** the
# temporal loop, and **re-compile** the variational form for each time step.
# In DOLFINx, a form is compiled whenever one calls {py:class}`dolfinx.fem.form`,
# or initialize a {py:class}`dolfinx.fem.petsc.LinearProblem`.

# ## Time-dependent constants
# To be able to use adaptive time stepping, we define `dt` as a {py:class}`dolfinx.fem.Constant`,
# such that we can re-assign values to the constant without having to re-compile the code.

dt = dolfinx.fem.Constant(mesh, 0.01)

# It is easy to re-assign a value to the constant by calling

dt.value = 0.005


# Similarly, we can define the diffusive coefficient `k` such as


# +
def k_func(t):
    return 0.1 if t < 0.5 else 0.05


k = dolfinx.fem.Constant(mesh, 0.1)
t = dolfinx.fem.Constant(mesh, 0.0)
while t.value < 1:
    # Update t
    t.value += dt.value
    # Update k
    k.value = k_func(t.value)
# We reset t
t.value = 0
# -

# ## Conditional values
# Next, we define the spatial and temporal source term using a
# {py:func}`ufl.conditional`
# We start by defining the spatially varying parameters of the mesh

x, y = ufl.SpatialCoordinate(mesh)

# Next, we create each component of the conditional statement

condition = ufl.lt(x, 0.5)
true_statement = 0.4 * y
false_statement = 0.5 * t
f = ufl.conditional(condition, true_statement, false_statement)

# Inside the compile code, this condition will be evaluated at every quadrature point.

# Now we have defined all varying functions outside the time loop, and each of them can be updated
# by updating the constant values

# ## Time-derivatives
# We use a simple backward Euler time stepping scheme to solve the problem.

u = ufl.TrialFunction(V)
u_n = dolfinx.fem.Function(V)
dudt = (u - u_n) / dt

# ## Full variational formulation
# With all the definitions above we can write the full variational form as
# $F(u, v)$

v = ufl.TestFunction(V)
dx = ufl.Measure("dx", domain=mesh)
F = dudt * v * dx + k * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx - f * v * dx

# ## Automatic extraction of the linear and bi-linear form
# We can exploit UFL to extract the linear and bi-linear components of the variational form

a, L = ufl.system(F)

# ## Explicit code generation
# We generate and compile the C code for these expressions using
# {py:func}`dolfinx.fem.form`

# +
a_compiled = dolfinx.fem.form(a)
L_compiled = dolfinx.fem.form(L)
# -

# ## Initial conditions
# We generate the initial condition by using lambda expressions


# +
def u_init(x, sigma=0.1, mu=0.3):
    """
    The input function x is a (3, number_of_points) numpy array, which is then
    evaluated with the vectorized numpy functions for efficiency
    """
    return (
        1.0
        / (2 * np.pi * sigma)
        * np.exp(-0.5 * ((x[0] - mu) / sigma) ** 2)
        * np.exp(-0.5 * ((x[1] - mu) / sigma) ** 2)
    )


u_n.interpolate(u_init)
# -

# + [markdown]
# ## Setting up the linear solver
# As we have now defined `a` and `L`, we are ready to set up a solver for the arising linear system.
# We use PETSc for this, and a direct solver.
# -

uh = dolfinx.fem.Function(V)
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
problem = dolfinx.fem.petsc.LinearProblem(
    a_compiled, L_compiled, u=uh, bcs=[], petsc_options=petsc_options, petsc_options_prefix="heat_"
)

# ## The temporal loop

# For each temporal step, we update the time variable and call the
# {py:meth}`solve<dolfinx.fem.petsc.LinearProblem.solve>` command that re-assemble the system

T = 1
while t.value < T:
    t.value += dt.value
    k.value = k_func(t.value)
    problem.solve()
    # Update previous solution
    u_n.x.array[:] = uh.x.array


# ```{note}
# Note that there is no definitions of variables within the temporal loop. There is simply copying of data or accumulation of data
# from the previous time step. This is a very efficient way of solving time-dependent problems in DOLFINx.
# ```
