# # Projections
#
# In [the introduction](./introduction), we considered the projection problem $u=\frac{f}{g}$ where $f$ and $g$ were some known functions.
# However, there are many cases where derived quantities are of interest, including stress calculations such as von Mises stresses.
#
# There are many ways of computing such quantities, and we have already covered how to set up a projection scheme
# Find $u\in V_h$ such that
# \begin{align}
#   \int_\Omega u\cdot v~\mathrm{d}x = \int_\Omega h(g,f,x)\cdot v~\mathrm{d}x\qquad \forall v \in V_h.
# \end{align}
# where $g$ and $f$ are functions from some finite element space.
#
#
# ## Reusable projector
# Imagine we want to solve a sequence of such post-processing steps for functions $f$ and $g$. If the mesh isn't changing between each projection,
# the left hand side is constant. Thus, it would make sense to only assemble the matrix once.
# Following this, we create a general projector class

import pyvista
import numpy as np
import dolfinx
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from typing import Optional


class Projector():
    """
    Projector for a given function.
    Solves Ax=b, where

    .. highlight:: python
    .. code-block:: python

        u, v = ufl.TrialFunction(Space), ufl.TestFunction(space)
        dx = ufl.Measure("dx", metadata=metadata)
        A = inner(u, v) * dx
        b = inner(function, v) * dx(metadata=metadata)

    Args:
        function: UFL expression of function to project
        space: Space to project function into
        petsc_options: Options to pass to PETSc
        jit_options: Options to pass to just in time compiler
        form_compiler_options: Options to pass to the form compiler
        metadata: Data to pass to the integration measure
    """

    _A: PETSc.Mat  # The mass matrix
    _b: PETSc.Vec  # The rhs vector
    _lhs: dolfinx.fem.FormMetaClass  # The compiled form for the mass matrix
    _ksp: PETSc.KSP  # The PETSc solver
    _x: dolfinx.fem.Function  # The solution vector
    _dx: ufl.Measure  # Integration measure

    def __init__(self,
                 space: dolfinx.fem.FunctionSpace,
                 petsc_options: Optional[dict] = None,
                 jit_options: Optional[dict] = None,
                 form_compiler_options: Optional[dict] = None,
                 metadata: Optional[dict] = None):
        petsc_options = {} if petsc_options is None else petsc_options
        jit_options = {} if jit_options is None else jit_options
        form_compiler_options = {} if form_compiler_options is None else form_compiler_options

        # Assemble projection matrix once
        u = ufl.TrialFunction(space)
        v = ufl.TestFunction(space)
        self._dx = ufl.Measure("dx", domain=space.mesh, metadata=metadata)
        a = ufl.inner(u, v) * self._dx(metadata=metadata)
        self._lhs = dolfinx.fem.form(a, jit_options=jit_options,
                                     form_compiler_options=form_compiler_options)
        self._A = dolfinx.fem.petsc.assemble_matrix(self._lhs)
        self._A.assemble()

        # Create vectors to store right hand side and the solution
        self._x = dolfinx.fem.Function(space)
        self._b = dolfinx.fem.Function(space)

        # Create Krylov Subspace solver
        self._ksp = PETSc.KSP().create(space.mesh.comm)
        self._ksp.setOperators(self._A)

        # Set PETSc options
        prefix = f"projector_{id(self)}"
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        self._ksp.setFromOptions()

        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(prefix)
        self._A.setFromOptions()
        self._b.vector.setOptionsPrefix(prefix)
        self._b.vector.setFromOptions()

    def reassemble_lhs(self):
        dolfinx.fem.petsc.assemble_matrix(self._A, self._lhs)
        self._A.assemble()

    def assemble_rhs(self, h: ufl.core.expr.Expr):
        """
        Assemble the right hand side of the problem
        """
        v = ufl.TestFunction(self._b.function_space)
        rhs = ufl.inner(h, v) * self._dx
        rhs_compiled = dolfinx.fem.form(rhs)
        self._b.x.array[:] = 0.
        dolfinx.fem.petsc.assemble_vector(self._b.vector, rhs_compiled)
        self._b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                                   mode=PETSc.ScatterMode.REVERSE)
        self._b.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                                   mode=PETSc.ScatterMode.FORWARD)

    def project(self, h: ufl.core.expr.Expr):
        """
        Compute projection using a PETSc KSP solver

        Args:
            assemble_rhs: Re-assemble RHS and re-apply boundary conditions if true
        """
        self.assemble_rhs(h)
        self._ksp.solve(self._b.vector, self._x.vector)
        return self._x

# With this class, we can send in any expression written in the unified form language to the projector,
# and then generate code for the right hand side assembly, and then solve the linear system.
# If we use LU factorization, most of the cost will be in the first projection, when the matrix is
# factorized. This is then cached, so that the solution algorithm is reduced to solving to linear problems;
# one upper diagonal matrix and one lower diagonal matrix.

# ## Approximation with continuous and discontinuous finite elements
#
# We will try to approximate the following function
#
# $h(x) = \begin{cases} \cos(\pi x) \quad\text{if } x<\frac{\pi}{5}\\
# -\sin(x) \quad\text{otherwise} \end{cases}
# We will use `ufl.conditional` as explained in the [the previous section](./form_compilation).


def h(mesh: dolfinx.mesh.Mesh):
    x = ufl.SpatialCoordinate(mesh)
    return ufl.conditional(ufl.lt(x[0], 0.1*ufl.pi), ufl.cos(ufl.pi*x[0]), -ufl.sin(x[0]))

# # Error norms and interpolation of UFL-expressions

# We will also compute the $L^2(\Omega) error with the following error-norm computation


def error_L2(uh, u_ex, degree_raise=3):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = dolfinx.fem.FunctionSpace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = dolfinx.fem.Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = dolfinx.fem.Function(W)
    if isinstance(u_ex, dolfinx.fem.Function):
        u_ex_W.interpolate(u_ex)
    elif isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = dolfinx.fem.Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = dolfinx.fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = dolfinx.fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


# # Comparison of continuous and discontinuous Lagrange

# Let us now test the code for a first order Lagrange space

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 20)
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))


V_projector = Projector(
    V, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = V_projector.project(h(V.mesh))

print(error_L2(uh, h(V.mesh)))

pyvista.start_xvfb(1.0)

# Next, we plot the solution


def plot_1D_scalar_function(u: dolfinx.fem.Function):
    u_grid = pyvista.UnstructuredGrid(
        *dolfinx.plot.create_vtk_mesh(u.function_space))
    u_grid.point_data["u"] = u.x.array
    warped = u_grid.warp_by_scalar()
    plotter = pyvista.Plotter()
    plotter.add_mesh(u_grid, style="points", show_edges=True)
    plotter.add_mesh(warped, style="wireframe")
    plotter.show_axes()
    plotter.view_xz()
    plotter.show()


plot_1D_scalar_function(uh)

# We can now repeat the study for a DG-1 function

W = dolfinx.fem.FunctionSpace(mesh, ("DG", 1))


W_projector = Projector(
    W, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
wh = W_projector.project(h(W.mesh))

print(error_L2(wh, h(W.mesh)))

plot_1D_scalar_function(wh)


# We observe that both solutions overshoot and undershoot around the discontinuity
# Let us try this for a variety of mesh sizes

for N in [50, 100, 200]:
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    W = dolfinx.fem.FunctionSpace(mesh, ("DG", 1))
    V_projector = Projector(
        V, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = V_projector.project(h(V.mesh))
    W_projector = Projector(
        W, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    wh = W_projector.project(h(W.mesh))
    error_Lagrange = error_L2(uh, h(V.mesh))
    error_DG = error_L2(wh, h(W.mesh))
    print(f"h={1./N}: Continuous Lagrange error: {error_Lagrange:.2e} Discontinuous Lagrange error: {error_DG:.2e}")

# We plot the finest solution for each space

# Lagrange
plot_1D_scalar_function(uh)

# Discontinuous Lagrange
plot_1D_scalar_function(wh)

# We still see overshoots with either space. This is known as Gibbs phenomenom and is discussed in detail in {cite}`ZHANG2022Gibbs`.
# However, if we align the discontinuity with the mesh, we will observe something interesting


def h_aligned(mesh: dolfinx.mesh.Mesh):
    x = ufl.SpatialCoordinate(mesh)
    return ufl.conditional(ufl.lt(x[0], 0.2), ufl.cos(ufl.pi*x[0]), -ufl.sin(x[0]))


for N in [20, 40, 80]:
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    W = dolfinx.fem.FunctionSpace(mesh, ("DG", 1))
    V_projector = Projector(
        V, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = V_projector.project(h_aligned(V.mesh))
    W_projector = Projector(
        W, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    wh = W_projector.project(h_aligned(W.mesh))
    error_Lagrange = error_L2(uh, h(V.mesh))
    error_DG = error_L2(wh, h_aligned(W.mesh))
    print(f"h={1./N}: Continuous Lagrange error: {error_Lagrange:.2e} Discontinuous Lagrange error: {error_DG:.2e}")

# Lagrange
plot_1D_scalar_function(uh)

# Discontinuous Lagrange
plot_1D_scalar_function(wh)

# ## Exercises
#
# ### Von Mises stresses
# When working with linear elasticity, it is often common to consider Von Mises stresses, defined as:
# \begin{align}
#   \sigma_m &= \sqrt{\frac{3}{2}s:s}\\
#   s(u)&=\sigma(u) -\frac{1}{3}\mathrm{tr}(\sigma(u))I
# \end{align}
#
# 1. Implement a simple linear elastic beam (2D), where the right hand side is fixed to the wall and the beam is deformed in y-direction
# by gravity.
# 2. Compute the von Mises stresses with a projector
#
# ### Discussion exercises
# 3. If we choose displacement in a first order continuous Lagrange space, what space should we place $\sigma_s$ in?
# 4. Can we use interpolation to compute Von Mises stresses in a continuous Lagrange space?
# 5. Can we use interpolation to compute Von Mises stresses in a discontinuous Lagrange space?
#
# ### Further implementation exercises
# 6. Implement the suitable options derived from 3.-5.


# ## Bibliography
# ```{bibliography}
# :filter: cited and ({"src/approximations"} >= docnames)
# ```
