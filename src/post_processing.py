# # Post-processing of derived quantities
#
# Let us again consider the equations of linear elasticity on a unit square.
#
# We consider the boundary conditions
#
# \begin{align}
# u(0, y) &= u(1, y) = 0\\
# \sigma(u)\cdot n &= -p n \qquad \text{at } (x,0)\cup(x,1)\\
# \end{align}

# We start by defining two marker functions for the two subdomains

from IPython import embed
import pyvista
import dolfinx
from mpi4py import MPI
import ufl
import numpy as np


def left(x):
    return np.isclose(x[1], 0) | np.isclose(x[1], 1)


def left_right(x):
    return np.isclose(x[0], 0) | np.isclose(x[0], 1)

# and a convenience function that takes in the mesh, and returns a mesh tag with these facets marked


def create_meshtags(mesh, top_bottom_marker: int = 1, left_right_marker: int = 2):
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    t_b_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim-1, top_bottom)
    l_r_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim-1, left_right)
    values = np.hstack([np.full_like(t_b_facets, top_bottom_marker),
                        np.full_like(l_r_facets, left_right_marker)])
    marked_facets = np.hstack([t_b_facets, l_r_facets])
    sort_order = np.argsort(marked_facets)
    return dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1,
                                 marked_facets[sort_order], values[sort_order])

# and convenience functions for $\sigma$ and $\epsilon$


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u, lam=2, mu=200):
    return lam * ufl.div(u)*ufl.Identity(len(u)) + 2*mu*epsilon(u)

# ## Problem specification

# We define a function that takes in the degree of Lagrange elements we will to use for this problem


def elasticity(mesh: dolfinx.mesh.Mesh, degree: int) -> dolfinx.fem.Function:
    el = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree)
    V = dolfinx.fem.FunctionSpace(mesh, el)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p_D = dolfinx.fem.Constant(mesh, 15.0)
    n = ufl.FacetNormal(mesh)
    T = p_D * n
    f = dolfinx.fem.Constant(mesh, (0.0, 0.0))

    # Create boundary conditions
    tb_marker = 1
    lr_marker = 2
    facet_marker = create_meshtags(mesh,
                                   top_bottom_marker=tb_marker,
                                   left_right_marker=lr_marker)
    fixed_dofs = dolfinx.fem.locate_dofs_topological(
        V, mesh.topology.dim-1, facet_marker.find(lr_marker))
    fixed_u = dolfinx.fem.Constant(mesh, (0., 0.))
    bcs = [dolfinx.fem.dirichletbc(fixed_u, fixed_dofs, V)]

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx - ufl.dot(T, v) * ds(tb_marker)

    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs)
    return problem.solve()


# We start by computing the solution with linear Lagrange elements

domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 50, 50)
uh = elasticity(domain, 1)

# We visualize the solution with the following code
pyvista.start_xvfb()

# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = dolfinx.plot.create_vtk_mesh(
    uh.function_space)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
vtk_values = np.zeros((len(uh.x.array)//len(uh), 3), dtype=np.float64)
vtk_values[:, :len(uh)] = uh.x.array.reshape((geometry.shape[0], len(uh)))
grid["u"] = vtk_values
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
p.view_xy()
p.show()

# ## Von Mises Stresses
# We want to compute the Von Mises stresses
# \begin{align}
#   \sigma_m &= \sqrt{\frac{3}{2}s:s}\\
#   s(u)&=\sigma(u) -\frac{1}{3}\mathrm{tr}(\sigma(u))I
# \end{align}


def s(u):
    return sigma(u) - 1. / 3 * ufl.tr(sigma(u)) * ufl.Identity(len(u))


def sigma_m(u):
    return ufl.sqrt(3./2*ufl.inner(s(u), s(u)))


# ## Interpolation
# If we use a first order Lagrange space for $u$, then $\sigma(u)$ is a cell-wise constant (DG-0).
# In turn $\sigma_m$ will be a piecewise constant.
# We use `dolfinx.fem.Expression` to interpolate this expression into DG-0.
# As explained with the variational formulations in [Form compilation](./form_compilation), the ufl-expression
# return by `sigma_m` has to be converted into C-code, that can be evaluated at any point in any cell.

def show_stresses_DG0(uh):
    domain = uh.function_space.mesh
    Q = dolfinx.fem.FunctionSpace(domain, ("DG", 0))
    stress_expr = dolfinx.fem.Expression(
        sigma_m(uh), Q.element.interpolation_points())
    stresses = dolfinx.fem.Function(Q)
    stresses.interpolate(stress_expr)

    vtk_grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(domain))
    vtk_grid.cell_data["Von-Mises stresses"] = stresses.x.array
    p = pyvista.Plotter()
    p.add_mesh(vtk_grid)
    p.view_xy()
    p.show()


show_stresses_DG0(uh)


# We next refine the grid and plot the stresses again

finer_domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 250, 250)
uh_finer = elasticity(finer_domain, 1)
show_stresses_DG0(uh_finer)

finest_domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 500, 500)
uh_finest = elasticity(finest_domain, 1)
show_stresses_DG0(uh_finest)


# We try the same thing, but instead of interpolating into DG-0,
# we project into a first order Lagrange space

def show_stresses_P1(uh):
    domain = uh.function_space.mesh
    Q = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    a = ufl.inner(p, q) * ufl.dx
    L = ufl.inner(sigma_m(uh), q) * ufl.dx
    problem = dolfinx.fem.petsc.LinearProblem(a, L)
    stresses = problem.solve()

    vtk_grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(Q))
    vtk_grid.point_data["Von-Mises stresses"] = stresses.x.array
    p = pyvista.Plotter()
    p.add_mesh(vtk_grid)
    p.view_xy()
    p.show()


# We look at the three different refinements

show_stresses_P1(uh)
show_stresses_P1(uh_finer)
show_stresses_P1(uh_finest)

# # Exercise:
#
# 1. Discuss potential drawbacks with using a P1 finite element space
# 2. Is there any way of speeding up the projection above if we would call it again on the same mesh?
