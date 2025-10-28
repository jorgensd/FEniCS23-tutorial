# # Integration of different kinds of forms
# As we have seen in the section on [Integration measures](./benefits_of_curved_meshes),
# we have used {py:func}`dolfinx.fem.assemble_scalar` to compute scalar integrals in DOLFINx.
# However, in the subsection {ref}`variational_form` we observed that we need to compute a matrix $A$ and a vector $b$.
# How do we compute them in DOLFINx without using {py:class}`dolfinx.fem.petsc.LinearProblem`?
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

element = ("Lagrange", 3, (2,))

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

# ### Assembly on the curved mesh

# We then define the right hand side equation over the curved mesh

v = ufl.TestFunction(V)
f = dolfinx.fem.Constant(curved_mesh, (0, -9.81))
dx_curved = ufl.Measure("dx", domain=curved_mesh)
L = linear_form(f, v, dx_curved)

# We compile the form

L_compiled = dolfinx.fem.form(L)

# We can now assemble the right hand side vector using {py:func}`dolfinx.fem.petsc.assemble_vector`
# Additionally, we compute the runtime of the assembly by using the `time` module.
# Since we want to assemble a vector, we pre-define it as a {py:class}`dolfinx.fem.Function` from the function space of the
# test function, so that we can re-use it for repeated assemblies. Additionally, this takes care of some of
# the memory management when interfacing with PETSc.

# +
import dolfinx.fem.petsc
from time import perf_counter

b = dolfinx.fem.Function(V)
# -

# ```{admonition} Accumulation of values when assembling a vector
# :class: dropdown
# When we call {py:func}`assemble_vector<dolfinx.fem.petsc.assemble_vector>`,
# we will not zero out values already present in the vector.
# Instead, the new values will be added to the existing values.
# Call `b.x.array[:] = 0.0` to zero out the vector before assembling.
# ```

b.x.array[:] = 0.0
start = perf_counter()
dolfinx.fem.petsc.assemble_vector(b.x.petsc_vec, L_compiled)
b.x.scatter_reverse(dolfinx.la.InsertMode.add)
end = perf_counter()

# We can now compare the performance of assembling over the linear mesh with the same resolution:
# ### Assembly on linear grid
# Define the linear form

v_lin = ufl.TestFunction(V_lin)
f_lin = dolfinx.fem.Constant(linear_mesh, (0, -9.81))
dx_lin = ufl.Measure("dx", domain=linear_mesh)
L_lin = linear_form(f_lin, v_lin, dx_lin)
L_lin_compiled = dolfinx.fem.form(L_lin)

# Define and assemble the vector

b_lin = dolfinx.fem.Function(V_lin)
b_lin.x.array[:] = 0.0
start_lin = perf_counter()
dolfinx.fem.petsc.assemble_vector(b_lin.x.petsc_vec, L_lin_compiled)
b_lin.x.scatter_reverse(dolfinx.la.InsertMode.add)
end_lin = perf_counter()

# ### Comparison with curved assembly

# + tags=["hide-input"]
print(f"Linear time (b): {end_lin - start_lin:.2e} Curved/Linear={(end - start) / (end_lin - start_lin):.2e}")
# -
# We additionally check the estimated quadrature degree for each integral

from ufl.algorithms import expand_derivatives, estimate_total_polynomial_degree

print(f"Curved (b) estimate: {estimate_total_polynomial_degree(expand_derivatives(L))}")
print(f"Linear (b) estimate: {estimate_total_polynomial_degree(expand_derivatives(L_lin))}")

# We observe that the assembly time is faster on the linear mesh,
# even though the quadrature degree is the same for both integrals.
# This is due to the computation of the Jacobian inside the assembly kernel.
#
# What about assembling a matrix?

# ## Bilinear forms (matrices)
# As we have seen in {ref}`variational_form`,
# we additionally get a **sparse matrix** when we assemble a bilinear form?
#
# ```{admonition} Why is it a sparse matrix?
# :class: dropdown
# As the basis function $\phi_i$ only have local support within the elements sharing the entity
# they are associated to, we only get a few non-zero entries in each row of the matrix.
# ```
#

# We use a similar approach as above to assemble a matrix.
# We start by defining the bilinear form with a {py:class}`TestFunction<ufl.TestFunction>`
# and a {py:class}`TrialFunction<ufl.TrialFunction>`.


# We consider the following bi-linear form
#
# $$
# a(u,v) = \int_\Omega u \cdot v + \nabla u \cdot \nabla v ~\mathrm{d}x.
# $$


def bilinear_form(u, v, dx):
    return ufl.inner(u, v) * dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * dx


a = bilinear_form(ufl.TrialFunction(V), v, dx_curved)
a_compiled = dolfinx.fem.form(a)

# Next, as a sparse matrix is expensive to create, we use the
# {py:func}`dolfinx.fem.petsc.create_matrix` function to create a matrix
# which we can use multiple times if we want to assemble the matrix multiple times.
# ```{admonition} Accumulation of values when assembling a matrix
# :class: dropdown
# DOLFINx does not zero out existing values in the matrix when assembling,
# so if you assemble the matrix multiple times,
# the values will be added to the existing values.
# Call {py:meth}`A.zeroEntries()<petsc4py.PETSc.Mat.zeroEntries>`
# to zero out the matrix before assembling.
# ```
#

A = dolfinx.fem.petsc.create_matrix(a_compiled)
A.zeroEntries()

# Next we can assemble the matrix using
# {py:func}`dolfinx.fem.petsc.assemble_matrix`

start_A = perf_counter()
dolfinx.fem.petsc.assemble_matrix(A, a_compiled)
A.assemble()
end_A = perf_counter()

# We do a similar computation for the linear mesh

a_lin = bilinear_form(ufl.TrialFunction(V_lin), v_lin, dx_lin)
a_lin_compiled = dolfinx.fem.form(a_lin)
A_lin = dolfinx.fem.petsc.create_matrix(a_lin_compiled)
A_lin.zeroEntries()
start_A_lin = perf_counter()
dolfinx.fem.petsc.assemble_matrix(A_lin, a_lin_compiled)
A_lin.assemble()
end_A_lin = perf_counter()

# We compare the assembly times

# + tags=["hide-input"]
print(
    f"Linear time (A): {end_A_lin - start_A_lin:.2e} Curved/Linear={(end_A - start_A) / (end_A_lin - start_A_lin):.2e}"
)
print(f"Linear time (b): {end_lin - start_lin:.2e} Curved/Linear={(end - start) / (end_lin - start_lin):.2e}")
# -

# We observe that assembling the matrix is two orders of magnitude slower than assembling the vector.
# We also observe that the assembly on a linear grid is faster than on a curved grid.

# We also compare the estimated quadrature degree of the integrals

# + tags=["hide-input"]
print(f"Curved (A) estimate: {estimate_total_polynomial_degree(expand_derivatives(a))}")
print(f"Linear (A) estimate: {estimate_total_polynomial_degree(expand_derivatives(a_lin))}")
# -
