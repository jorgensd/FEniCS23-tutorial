# # Integration of different kinds of forms
# As we have seen above, we have used `dolfinx.fem.assemble_scalar` to
# assemble scalar valued expressions in DOLFINx.
# However, in {ref}`variational_form` we observed that we need to compute a matrix $A$ and a vector $b$.
# How do we compute them in DOLFINx without using `dolfinx.fem.petsc.LinearProblem`?
#
# In this section, we will compare assembly on a linear and a curved mesh.
# We start by creating these two meshes with the function from [the previous section](./benefits_of_curved_meshes).

# + tags=["remove-output"]
from benefits_of_curved_meshes import generate_mesh
resolution = 0.01
linear_mesh, _, _ = generate_mesh(resolution, 1)
curved_mesh, _, _ = generate_mesh(resolution, 3)
# -

# For this section, we will use the following finite element definition

element = ("Lagrange", 3, (2, ))

# and the function spaces

import dolfinx
V = dolfinx.fem.functionspace(curved_mesh, element)
V_lin = dolfinx.fem.functionspace(linear_mesh, element)

# ## Assembling a vector
# Then, we will consider the right hand side of the variational form:
#
# $$
# \int_\Omega f \cdot v ~\mathrm{d}x\qquad \forall v \in V.
# $$
#
# We make a convenience function to create this form

# +
import ufl


def linear_form(f, v, dx):
    return ufl.inner(f, v) * dx
# -

# We then define the right hand side equation over the curved mesh

v = ufl.TestFunction(V)
f = dolfinx.fem.Constant(curved_mesh, (0, -9.81))
dx_curved = ufl.Measure("dx", domain=curved_mesh)
L = dolfinx.fem.form(linear_form(f, v, dx_curved))

# We can now assemble the right hand side vector using `dolfinx.fem.petsc.assemble_vector`
# Additionally, we compute the runtime of the assembly by using the `time` module

from petsc4py import PETSc
import dolfinx.fem.petsc
from time import perf_counter

# We create a vector in the function space of the test function, that we can assemble data into

b = dolfinx.fem.Function(V)

# ```{note}
# When we call `assemble_vector`, we will not zero out values already present in the vector.
# Instead, the new values will be added to the existing values.
# Call `b.x.array[:] = 0.0` to zero out the vector before assembling.
# ```

b.x.array[:] = 0.0
start = perf_counter()
dolfinx.fem.petsc.assemble_vector(b.x.petsc_vec, L)
b.x.scatter_reverse(dolfinx.la.InsertMode.add)
end = perf_counter()

# We can now compare the performance of assembling over the linear mesh with the same resolution:

# 1. Define the linear form

v_lin = ufl.TestFunction(V_lin)
f_lin = dolfinx.fem.Constant(linear_mesh, (0, -9.81))
dx_lin = ufl.Measure("dx", domain=linear_mesh)
L_lin = dolfinx.fem.form(linear_form(f_lin, v_lin, dx_lin))

# 2. Assemble the vector

b_lin = dolfinx.fem.Function(V_lin)
b_lin.x.array[:] = 0.0
start_lin = perf_counter()
dolfinx.fem.petsc.assemble_vector(b_lin.x.petsc_vec, L_lin)
b_lin.x.scatter_reverse(dolfinx.la.InsertMode.add)
end_lin = perf_counter()

# 3. Compare with curved assembly

print(f"Linear time (b): {end_lin-start_lin:.2e} Curved/Linear={(end-start)/(end_lin-start_lin):.2e}")

# We observe that the assembly time is only slight faster on the linear mesh.
# What about assembling a matrix?

# ## Bilinear forms (matrices)
# As we have seen in {ref}`variational_form`, we additionally get a **sparse matrix** when we assemble a bilinear form?
#
# ### Question:
# - Why is it a sparse matrix?

# We use a similar approach as above to assemble a matrix.
# We start by defining the bilinear form with a `TestFunction` and a `TrialFunction`.


# We consider the following bi-linear form

def bilinear_form(u, v, dx):
    return ufl.inner(u, v) * dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * dx


u = ufl.TrialFunction(V)
a = dolfinx.fem.form(bilinear_form(u,v, dx_curved))

# Next, as a sparse matrix is expensive to create, we use the `dolfinx.fem.petsc.create_matrix` function to create a matrix
# which we can use multiple times if we want to assemble the matrix multiple times.
# ```{note}
# DOLFINx does not zero out existing values in the matrix when assembling, so if you assemble the matrix multiple times,
# the values will be added to the existing values.
# Call `A.zeroEntries()` to zero out the matrix before assembling.
# ```
#

A = dolfinx.fem.petsc.create_matrix(a)
A.zeroEntries()

# Next we can assemble the matrix using `dolfinx.fem.petsc.assemble_matrix`

start_A = perf_counter()
dolfinx.fem.petsc.assemble_matrix(A, a)
A.assemble()
end_A = perf_counter()

# We do a similar computation for the linear mesh

u_lin = ufl.TrialFunction(V_lin)
a_lin = dolfinx.fem.form(bilinear_form(u_lin, v_lin, dx_lin))
A_lin = dolfinx.fem.petsc.create_matrix(a_lin)
A_lin.zeroEntries()
start_A_lin = perf_counter()
dolfinx.fem.petsc.assemble_matrix(A_lin, a_lin)
A_lin.assemble()
end_A_lin = perf_counter()

# We compare the assembly times

print(f"Linear time (A): {end_A_lin-start_A_lin:.2e} Curved/Linear={(end_A-start_A)/(end_A_lin-start_A_lin):.2e}")

# We observe that assembling the matrix is two orders of magnitude slower than assembling the vector.
# We also observe that the assembly on a linear grid is faster than on a curved grid.

