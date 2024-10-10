# # Approximation with continuous and discontinuous finite elements
#
# We introduced the notion of a projection in {ref}`variational_form` where we want to find the best
# approximation of an expression in a finite element space.
#
# The goal of this section is to approximate
#
# $$
# h(x) = \begin{cases} \cos(\pi x) \quad\text{if } x<\alpha\\
# -\sin(x) \quad\text{otherwise} \end{cases}
# $$
#
# where $\alpha$ is a pre-defined constant.
# We will use `ufl.conditional` as explained in the [the previous section](./form_compilation).

# +
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc
import ufl

def h(alpha, mesh: dolfinx.mesh.Mesh):
    x = ufl.SpatialCoordinate(mesh)
    return ufl.conditional(ufl.lt(x[0], alpha), ufl.cos(ufl.pi * x[0]), -ufl.sin(x[0]))
# -

# ## Reusable projector
# Imagine we want to solve a sequence of such post-processing steps for functions $f$ and $g$.
# If the **mesh is not changing** between each projection, the left hand side is constant.
# Thus, it would make sense to assemble the matrix **once**.
# Following this, we create a general projector class (`class Projector`)

# + tags=["hide-input"]
from typing import Optional

import numpy as np
import pyvista

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
# -

# With this class, we can send in any expression written in {term}`UFL` to the projector,
# and then generate code for the right hand side assembly, and then solve the linear system.
# If we use **LU factorization**, most of the cost will be in the first projection, when the matrix is
# factorized. This is then cached, so that the solution algorithm is reduced to solving to linear problems;
# one upper diagonal matrix and one lower diagonal matrix.

# ## Non-aligning discontinuity
# We will start considering the case where $\alpha$ is not aligned with the mesh.
# We choose $\alpha = \frac{\pi}{10}$ and get the following $h$:
#
# $$
# h(x) = \begin{cases} \cos(\pi x) \quad\text{if } x<\frac{\pi}{10}\\
# -\sin(x) \quad\text{otherwise} \end{cases}
# $$

# ````{admonition} The partial-function
# :class: dropdown
# For the next cases, we would like to fix alpha in our function `h`, but we want to keep the mesh as a variable.
# We can use the [partial](https://docs.python.org/3/library/functools.html#functools.partial)
# function from the `functools` module to fix the first argument of a function.
# It creates a new object that behaves as a function where `mesh` is the only input parameter.
# i.e.
# ```python
# from functools import partial
# def f(x, y, z):
#   print(f"{x=}, {y=}, {z=}")
# h = partial(f, 3, 4)
# h(7) # prints x=3, y=4, z=7
# ```
# ````

from functools import partial
h_nonaligned = partial(h, np.pi / 10)

# Let us now try to use the re-usable projector to approximate this function
# with a **continuous Lagrange space of order 1**

# +
Nx = 20
mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, Nx)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
V_projector = Projector(V, petsc_options=petsc_options)
uh = V_projector.project(h_nonaligned(V.mesh))
# -

# We can now repeat the study for a **discontinuous Lagrange space of order 1**

W = dolfinx.fem.functionspace(mesh, ("DG", 1))
W_projector = Projector(W, petsc_options=petsc_options)
wh = W_projector.project(h_nonaligned(W.mesh))

# We compare the two solutions side by side

# + tags=["hide-input"]
def warp_1D(u: dolfinx.fem.Function, factor=1):
    """Convenience function to warp a 1D function for visualization in pyvista"""
    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))
    u_grid.point_data["u"] = u.x.array
    return u_grid.warp_by_scalar(factor=factor)


def create_side_by_side_plot(u_continuous:dolfinx.fem.Function, u_dg:dolfinx.fem.Function, ):
    def num_glob_cells(u:dolfinx.fem.Function)->int:
        mesh = u.function_space.mesh
        cell_map = mesh.topology.index_map(mesh.topology.dim)
        return cell_map.size_global

    pyvista_continuous = warp_1D(u_continuous)
    pyvista_dg = warp_1D(u_dg)

    pyvista.set_jupyter_backend("static")
    plotter = pyvista.Plotter(shape=(1, 2))
    plotter.subplot(0,0)
    plotter.add_text(f"Continuous Lagrange N={num_glob_cells(u_continuous)}")
    plotter.add_mesh(pyvista_continuous, style="wireframe", line_width=3)
    plotter.show_axes()
    plotter.view_xz()
    plotter.subplot(0,1)
    plotter.add_text(f"Discontinuous Lagrange N={num_glob_cells(u_dg)}")
    plotter.add_mesh(pyvista_dg, style="wireframe", line_width=3)
    plotter.show_axes()
    plotter.view_xz()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    pyvista.set_jupyter_backend("html")


pyvista.start_xvfb(1.0)
create_side_by_side_plot(uh, wh)
# -

# We observe that both solutions overshoot and undershoot around the discontinuity
# Let us refine the mesh several times to see if the solution converges

# + tags=["hide-input"]
for N in [50, 100, 200]:
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    W = dolfinx.fem.functionspace(mesh, ("DG", 1))
    V_projector = Projector(V, petsc_options=petsc_options)
    uh = V_projector.project(h_nonaligned(V.mesh))
    W_projector = Projector(W, petsc_options=petsc_options)
    wh = W_projector.project(h_nonaligned(W.mesh))
    create_side_by_side_plot(uh, wh)
# -

# We still see overshoots with either space. This is known as Gibbs phenomenon and is
# discussed in detail in {cite}`ZHANG2022Gibbs`.

# ## Grid-aligned discontinuity
# Next, we choose $\alpha = 0.2$ and choose grid sizes such that the discontinuity is aligned with a cell boundary.

h_aligned = partial(h, 0.2)

# + tags=["hide-input"]
for N in [20, 40, 80]:
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    W = dolfinx.fem.functionspace(mesh, ("DG", 1))
    V_projector = Projector(V, petsc_options=petsc_options)
    uh = V_projector.project(h_aligned(V.mesh))
    W_projector = Projector(W, petsc_options=petsc_options)
    wh = W_projector.project(h_aligned(W.mesh))

    create_side_by_side_plot(uh, wh)
# -

# ## Interpolation of functions and UFL-expressions

# Above we have defined a function that is dependent on the spatial coordinates of `ufl`, and it is a **purely symbolic expression**.
# If we want to evaluate this expression, either at a given point or interpolate it into a function space, we need to compile code
# similar to the code generated with `dolfinx.fem.form` or calling FFCx.
# The main difference is that for an expression, there is **no summation over quadrature**.
# To perform this compilation for a given point in the reference cell, we call


mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 7)
compiled_h = dolfinx.fem.Expression(h_aligned(mesh), np.array([0.5]))

# We can now evaluate the expression at the point 0.5 in the reference element for any cell
# (this coordinate is then pushed forward to the given input cell).
# For instance, we can evaluate this expression in the cell with index 0 with

compiled_h.eval(mesh, np.array([0], dtype=np.int32))

# ## Interpolate expressions
# We can also use expressions for post-processing, by interpolating into an appropriate
# finite element function space (`Q`). To do so, we compile the UFL-expression to be evaluated
# at the interpolation points of `Q`.

# +
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
u = dolfinx.fem.Function(V)

Q = dolfinx.fem.functionspace(mesh, ("DG", 1))

dudx = ufl.grad(u)[0]
compile_dudx = dolfinx.fem.Expression(dudx, Q.element.interpolation_points())
# -

# We populate `u` with some data on some part of the domain

left_cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, lambda x: x[0]>=0.3+1e-14)
u.interpolate(lambda x: x[0]**2, cells=left_cells)

# We can then interpolate `dudx` into `Q` with

q = dolfinx.fem.Function(Q)
q.interpolate(compile_dudx)

# and plot the result

# + tags=["hide-input"]
pyvista.set_jupyter_backend("static")
plotter = pyvista.Plotter()
plotter.add_mesh(warp_1D(u, 1), style="wireframe", line_width=5)
plotter.view_xz()
if not pyvista.OFF_SCREEN:
    plotter.show()
plotter = pyvista.Plotter()
plotter.add_mesh(warp_1D(q, 0.1), style="wireframe", line_width=5)
plotter.show_axes()
plotter.view_xz()
if not pyvista.OFF_SCREEN:
    plotter.show()
pyvista.set_jupyter_backend("html")
# -

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
