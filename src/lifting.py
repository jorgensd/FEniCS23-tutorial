# # Application of Dirichlet boundary conditions
# In this section, we will cover how one applies strong Dirichlet conditions to a variational problem.
#
# We consider the equations of linear elasticity,
#
# \begin{align}
# -\nabla \cdot \sigma (u) &= f && \text{in } \Omega\\
# \sigma(u)&= \lambda \mathrm{tr}(\epsilon(u))I + 2 \mu \epsilon(u)\\
# \epsilon(u) &= \frac{1}{2}\left(\nabla u + (\nabla u )^T\right)
# \end{align}
# where $\sigma$ is the stress tensor, $f$ is the body force per unit volume,
# $\lambda$ and $\mu$ are Lamé's elasticity parameters for the material in $\Omega$,
# $I$ is the identity tensor, $\mathrm{tr}$ is the trace operator on a tensor,
# $\epsilon$ is the symmetric strain tensor (symmetric gradient),
# and $u$ is the displacement vector field. Above we have assumed isotropic elastic conditions.
#
# We will use model the equations on a beam, with a fixed displacement at the right end,
# clamping at the right end and no traction boundary conditions on all other boundaries.

from IPython import embed
from mpi4py import MPI
import dolfinx
import numpy as np
import ufl
from petsc4py import PETSc

L, W, H, = 10., 3., 3.
mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD, [[0., 0., 0.], [L, W, H]],
    [15, 7, 7], cell_type=dolfinx.mesh.CellType.hexahedron)
tdim = mesh.topology.dim

el = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
V = dolfinx.fem.FunctionSpace(mesh, el)

# ## Define boundaries
# We start by locate the various facets for the different boundary conditions.
# First, we find all boundary facets (those facets that are connected to only one cell)

mesh.topology.create_connectivity(tdim-1, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

# Next we find those facets that should be clamped, and those that should have a non-zero traction on it.

clamped_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, tdim-1,
    lambda x: np.isclose(x[0], 0.))
prescribed_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, tdim-1,
    lambda x: np.isclose(x[0], L))

# As all mesh entities are represented as integers, we can find the boundary facets by
# remaining facets using numpy set operations


free_facets = np.setdiff1d(
    boundary_facets,
    np.union1d(clamped_facets, prescribed_facets))

# Next, we can define a meshtag object for all the facets in the mesh

num_facets = mesh.topology.index_map(tdim-1).size_local
markers = np.zeros(num_facets, dtype=np.int32)
markers[clamped_facets] = 1
markers[prescribed_facets] = 2
markers[free_facets] = 3
facet_marker = dolfinx.mesh.meshtags(
    mesh, tdim-1,
    np.arange(num_facets, dtype=np.int32), markers)


# # Variational formulation
x = ufl.SpatialCoordinate(mesh)
T_0 = dolfinx.fem.Constant(mesh, (0., 0., 0.))
E = dolfinx.fem.Constant(mesh, 1.4e3)
nu = dolfinx.fem.Constant(mesh, 0.3)
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
f = dolfinx.fem.Constant(mesh, (0., 0., 0.))


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))


ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(T_0, v) * ds(3)
a_compiled = dolfinx.fem.form(a)
L_compiled = dolfinx.fem.form(L)

# ## Dirichlet conditions
# We start by finding all dofs associated with the facets marked with each of the Dirichlet conditions

clamped_dofs = dolfinx.fem.locate_dofs_topological(
    V, tdim-1, clamped_facets)
displaced_dofs = dolfinx.fem.locate_dofs_topological(
    V, tdim-1, prescribed_facets)

# Next, we define the prescribed displacement
# $u_D(L,y,z)=(0,0,W/4)$

u_prescribed = dolfinx.fem.Constant(mesh, (0., 0., H/4))
u_clamped = dolfinx.fem.Constant(mesh, (0., 0.0, 0.))

# We define the Dirichlet boundary condition object as

bcs = [dolfinx.fem.dirichletbc(u_clamped, clamped_dofs, V),
       dolfinx.fem.dirichletbc(u_prescribed, displaced_dofs, V)]

# In most situations, you would just pass this object to the linear problem and it would be handled for you
petsc_options = {"ksp_type": "preonly",
                 "pc_type": "lu", "pc_factor_solver_type": "mumps"}
problem = dolfinx.fem.petsc.LinearProblem(
    a, L, bcs=bcs, petsc_options=petsc_options)
u = problem.solve()
u.x.scatter_forward()
with dolfinx.io.VTXWriter(mesh.comm, "u_ref.bp", [u]) as vtx:
    vtx.write(0.0)

# However, what goes on under the hood?

# ### Identity row approach
# If we would not supply any boundary conditions to the system, we would have a matrix

A = dolfinx.fem.petsc.assemble_matrix(a_compiled)
A.assemble()

# which is symmetric due to the symmetry of the stress tensor

print(f"Matrix A is symmetric after assembly: {A.isSymmetric(1e-5)}")

# Let us split the degrees of freedom into two disjoint sets, $u_d$, and $u_{bc}$, and set up the corresponding linear system
# \begin{align}
# \begin{pmatrix}
# A_{d,d} & A_{d, bc} \\
# A_{bc,d} & A_{bc, bc}
# \end{pmatrix}
# \begin{pmatrix}
# u_d \\
# u_{bc}
# \end{pmatrix}
# &=
# \begin{pmatrix}
# b_d \\
# b_{bc}
# \end{pmatrix}
# \end{align}
# We can now reduce the system to
# \begin{align}
# \begin{pmatrix}
# A_{d,d} & A_{d, bc} \\
# 0 & I
# \end{pmatrix}
# \begin{pmatrix}
# u_d \\
# u_{bc}
# \end{pmatrix}
# &=
# \begin{pmatrix}
# b_d \\
# g
# \end{pmatrix}
# \end{align}
# where $g$ is the vector satisfying the various Dirichlet conditions.

# We do this with DOLFINx in the following way

for bc in bcs:
    dofs, _ = bc.dof_indices()
    A.zeroRowsLocal(dofs, diag=1)

# Next, we could assemble the RHS vector as normal and set the BC values

b = dolfinx.fem.petsc.assemble_vector(L_compiled)
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.petsc.set_bc(b, bcs)
b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
              mode=PETSc.ScatterMode.FORWARD)

# Then we could define a PETSc Krylov subspace solver
uh = dolfinx.fem.Function(V)
solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.getPC().setFactorSolverType("mumps")
solver.solve(b, uh.vector)
uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                      mode=PETSc.ScatterMode.FORWARD)

# We can check that we get the same solution

assert np.allclose(u.x.array, uh.x.array)


# However, the matrix `A` is no longer symmetric, meaning that we exclude a whole class of iterative solvers
# (Conjugate Gradient)

print(f"Matrix A is symmetric after bc application: {A.isSymmetric(1e-5)}")

# ### Lifting
# Lifting is a procedure that can be use to reduce the number of unknowns in the system, to only be those dofs not constrained by
# the Dirichlet condition.
# We use the same split as we did in the previous approach
# \begin{align}
# \begin{pmatrix}
# A_{d,d} & A_{d, bc} \\
# A_{bc,d} & A_{bc, bc}
# \end{pmatrix}
# \begin{pmatrix}
# u_d \\
# u_{bc}
# \end{pmatrix}
# &=
# \begin{pmatrix}
# b_d \\
# b_{bc}
# \end{pmatrix}
# \end{align}
# However, as we know what $u$ is at $u_bc$, we can eliminate all these unknowns from the system
# \begin{align}
# A_{d,d}
# u_d
# &=
# b_d
# -
# A_{d, bc}g
# \end{align}
# and we end up with a smaller system, which is symmetric if $A_{d,d}$ is symmetric.
# However, we do not want to remove the degrees of freedom from the sparse, matrix, as it makes the matrix
# less adaptible for varying boundary conditions, so we set up the system
# \begin{align}
# \begin{pmatrix}
# A_{d,d} & 0 \\
# 0 & I
# \end{pmatrix}
# \begin{pmatrix}
# u_d \\
# u_{bc}
# \end{pmatrix}
# &=
# \begin{pmatrix}
# b_d - A_{d,bc}g \\
# g
# \end{pmatrix}
# \end{align}
# We do this in DOLFINx with the following commands
#
# 1. `assemble_matrix` with Dirichlet conditions

A_lifting = dolfinx.fem.petsc.assemble_matrix(a_compiled, bcs=bcs)
A_lifting.assemble()

# converts all columns and rows with a Dirichlet dofs to the identity row/column (during local assembly.
# 2. `assemble_vector` without DirichletBC

b_lifting = dolfinx.fem.petsc.assemble_vector(L_compiled)

# 3. `apply_lifting`, i.e subtract $A_{d,bc}g$ from the vector

dolfinx.fem.petsc.apply_lifting(b_lifting, [a_compiled], bcs=[bcs])
b_lifting.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)

# 4. `set_bc`, i.e set the degrees of freedom for known dofs

dolfinx.fem.petsc.set_bc(b_lifting, bcs)
b_lifting.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                      mode=PETSc.ScatterMode.FORWARD)

print(
    f"Matrix is symmetric after lifting assembly: {A_lifting.isSymmetric(1e-5)}")

solver.setOperators(A_lifting)
u_lifted = dolfinx.fem.Function(V)
solver.solve(b_lifting, u_lifted.vector)
u_lifted.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                            mode=PETSc.ScatterMode.FORWARD)

assert np.allclose(u_lifted.x.array, uh.x.array)
