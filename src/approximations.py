# # Projection and interpolation
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

from typing import Optional

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pyvista

import dolfinx
import dolfinx.fem.function
import dolfinx.fem.petsc
import ufl


class Projector:
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
    _lhs: dolfinx.fem.Form  # The compiled form for the mass matrix
    _ksp: PETSc.KSP  # The PETSc solver
    _x: dolfinx.fem.Function  # The solution vector
    _dx: ufl.Measure  # Integration measure

    def __init__(
        self,
        space: dolfinx.fem.FunctionSpace,
        petsc_options: Optional[dict] = None,
        jit_options: Optional[dict] = None,
        form_compiler_options: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        petsc_options = {} if petsc_options is None else petsc_options
        jit_options = {} if jit_options is None else jit_options
        form_compiler_options = {} if form_compiler_options is None else form_compiler_options

        # Assemble projection matrix once
        u = ufl.TrialFunction(space)
        v = ufl.TestFunction(space)
        self._dx = ufl.Measure("dx", domain=space.mesh, metadata=metadata)
        a = ufl.inner(u, v) * self._dx(metadata=metadata)
        self._lhs = dolfinx.fem.form(a, jit_options=jit_options, form_compiler_options=form_compiler_options)
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
        for opt in opts.getAll().keys():
            del opts[opt]

        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(prefix)
        self._A.setFromOptions()
        self._b.x.petsc_vec.setOptionsPrefix(prefix)
        self._b.x.petsc_vec.setFromOptions()

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
        self._b.x.array[:] = 0.0
        dolfinx.fem.petsc.assemble_vector(self._b.x.petsc_vec, rhs_compiled)
        self._b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self._b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    def project(self, h: ufl.core.expr.Expr) -> dolfinx.fem.Function:
        """
        Compute projection using a PETSc KSP solver

        Args:
            assemble_rhs: Re-assemble RHS and re-apply boundary conditions if true
        """
        self.assemble_rhs(h)
        self._ksp.solve(self._b.x.petsc_vec, self._x.x.petsc_vec)
        return self._x

    def __del__(self):
        self._A.destroy()
        self._ksp.destroy()


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
    return ufl.conditional(ufl.lt(x[0], 0.1 * ufl.pi), ufl.cos(ufl.pi * x[0]), -ufl.sin(x[0]))


# # Interpolation of functions and  UFL-expressions

# Above we have defined a function that is dependent on the spatial coordinates of `ufl`, and it is a purely symbolic expression.
# If we want to evaluate this expression, either at a given point or interpolate it into a function space, we need to compile code
# similar to the code generated with `dolfinx.fem.form` or calling FFCx. The main difference is that for an expression, there is no summation
# over quadrature. To perform this compilation for a given point in the reference cell, we call


mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 20)
compiled_h = dolfinx.fem.Expression(h(mesh), np.array([0.5]))

# We can now evaluate the expression at the point 0.5 in the reference element for any cell (this coordinate is then pushed forward to the given input cell).
# For instance, we can evaluate this expression in the cell with index 0 with
compiled_h.eval(mesh, np.array([0], dtype=np.int32))

# We can also interpolate functions from an expression into any suitable function space by calling

V = dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", 2))
compiled_h_for_V = dolfinx.fem.Expression(h(mesh), V.element.interpolation_points())

# # Comparison of continuous and discontinuous Lagrange

# Let us now test the code for a first order Lagrange space

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 20)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


V_projector = Projector(V, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = V_projector.project(h(V.mesh))

pyvista.start_xvfb(1.0)

# Next, we plot the solution


def plot_1D_scalar_function(u: dolfinx.fem.Function, title: str):
    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))
    u_grid.point_data["u"] = u.x.array
    warped = u_grid.warp_by_scalar()
    plotter = pyvista.Plotter()
    plotter.add_title(title, font_size=12)
    plotter.add_mesh(warped, style="wireframe")
    plotter.show_axes()
    plotter.view_xz()
    plotter.show()


plot_1D_scalar_function(uh, "First order Lagrange")

# We can now repeat the study for a DG-1 function

W = dolfinx.fem.functionspace(mesh, ("DG", 1))


W_projector = Projector(W, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
wh = W_projector.project(h(W.mesh))


plot_1D_scalar_function(wh, "First order Discontinuous Lagrange")


# We observe that both solutions overshoot and undershoot around the discontinuity
# Let us try this for a variety of mesh sizes

for N in [50, 100, 200]:
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    W = dolfinx.fem.functionspace(mesh, ("DG", 1))
    V_projector = Projector(V, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = V_projector.project(h(V.mesh))
    W_projector = Projector(W, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    wh = W_projector.project(h(W.mesh))
    # Lagrange
    plot_1D_scalar_function(uh, f"First order Lagrange ({N})")

    # Discontinuous Lagrange
    plot_1D_scalar_function(wh, f"First order discontinuous Lagrange ({N})")


# We still see overshoots with either space. This is known as Gibbs phenomenom and is discussed in detail in {cite}`ZHANG2022Gibbs`.
# However, if we align the discontinuity with the mesh, we will observe something interesting


def h_aligned(mesh: dolfinx.mesh.Mesh):
    x = ufl.SpatialCoordinate(mesh)
    return ufl.conditional(ufl.lt(x[0], 0.2), ufl.cos(ufl.pi * x[0]), -ufl.sin(x[0]))


for N in [20, 40, 80]:
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    W = dolfinx.fem.functionspace(mesh, ("DG", 1))
    V_projector = Projector(V, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = V_projector.project(h_aligned(V.mesh))
    W_projector = Projector(W, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    wh = W_projector.project(h_aligned(W.mesh))

    # Lagrange
    plot_1D_scalar_function(uh, f"Aligned first order Lagrange ({N})")

    # Discontinuous Lagrange
    plot_1D_scalar_function(wh, f"Aligned first order discontinuous Lagrange ({N})")

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
