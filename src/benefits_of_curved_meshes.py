# # Integration measures
# In this section we will cover how to compute different kinds of integrals in DOLFINx.
# To illustrate this, we will use the mesh from {ref}`mesh_generation:eshelby`. where we saw that it is possible
# to control the order of the underlying mesh elements using GMSH.
# In this section we will see how this can be used to improve the accuracy of the solution.
# We will also explore potential drawbacks.

# First we create an easily customizable function based on the previous example

from mpi4py import MPI
import dolfinx
import ufl
import gmsh
import numpy as np

aspect_ratio = 0.5
R_i = 0.3
R_e = 0.8
center = (0,0,0)


# + tags = ["hide-input"]
def generate_mesh(resolution: float, order:int):
    """Generate a mesh with a given minimal resolution of a given order."""
    assert order >= 1
    gmsh.initialize()
    gmsh.model.add("eshelby")

    inner_disk = gmsh.model.occ.addDisk(*center, R_i, aspect_ratio * R_i)
    outer_disk = gmsh.model.occ.addDisk(*center, R_e, R_e)

    _, map_to_input = gmsh.model.occ.fragment(
                [(2, outer_disk)], [(2, inner_disk)]
            )
    gmsh.model.occ.synchronize()

    circle_inner = [surface[1] for surface in map_to_input[1] if surface[0] == 2]
    circle_outer = [surface[1] for surface in map_to_input[0] if surface[0] == 2 and surface[1] not in circle_inner]

    gmsh.model.addPhysicalGroup(2, circle_inner, tag=3)
    gmsh.model.addPhysicalGroup(2, circle_outer, tag=7)

    inner_boundary = gmsh.model.getBoundary([(2, entity) for entity in circle_inner], recursive=False, oriented=False)
    outer_boundary = gmsh.model.getBoundary([(2, entity) for entity in circle_outer], recursive=False, oriented=False)

    interface = [boundary[1] for boundary in inner_boundary if boundary[0] == 1]
    ext_boundary = [boundary[1] for boundary in outer_boundary if boundary[0] == 1 and boundary[1] not in interface]

    gmsh.model.addPhysicalGroup(1, interface, tag=12)
    gmsh.model.addPhysicalGroup(1, ext_boundary, tag=15)


    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    mesh, cell_marker, facet_marker = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return mesh, cell_marker, facet_marker
# -

# ## Integration over cells
# We start by integrating over the cells of a (linear) mesh. For this, we can either use
# `ufl.dx`, which is a UFL integration measure that indicates that we want to integrate
# over all cells of a given domain.

linear_mesh, linear_celltags, linear_facettags = generate_mesh(0.5, 1)
dx = ufl.dx(domain=linear_mesh)

# We can now compute the area of the domain by integrating the constant function 1 over the domain.

one = ufl.constantvalue.IntValue(1)
area = one * dx

# We have above create a UFL form for the integral over the domain multiplied by a constant.
# We now want to assemble this over the mesh.
# ```{admonition} Assembly of a scalar value in parallel
# 1. Compile the form (generate code for the scalar form).
# 2. Compute the local contribution of each cell owned by the current process.
# 3. Accumulate the local contributions across all processes.
# ```

compiled_area = dolfinx.fem.form(area)
local_area = dolfinx.fem.assemble_scalar(compiled_area)
global_area = linear_mesh.comm.allreduce(local_area, op=MPI.SUM)

# We can compare this to the area of the exact geometry

A_ex = np.pi*R_e**2
print(f"Area of the domain is {global_area}, expected {A_ex}\n", 
      f"Relative error: {np.abs(global_area - A_ex)/A_ex*100:.2f}%")

# We observe a small error in the total area of the circle, but what about the area of each subdomain?


# ## Integration over subdomains
# In the mesh generation, we marked the ellipsoid with the value 3 and the remainder of the domain with the value 7.
# We use the other outputs from `dolfinx.io.gmshio.model_to_mesh` to extract to access the subdomain information.
# The second output of the function is a `dolfinx.mesh.MeshTags` object, consisting of a list of all cells in the
# `linear_mesh` and its corresponding marker.
# We can use this information within the a `ufl.Measure` to restrict the integration to a subdomain.

dx_with_data = ufl.Measure("dx", domain=linear_mesh, subdomain_data=linear_celltags)

# We can now create a form which only integrates over the cells marked with the value 3 with the following syntax
# ```{note}
# Remember that to call assemble on any form we need to multiply by an integration measure.
# ```

inner_area = dolfinx.fem.form(one*dx_with_data(3))

# We can also pass multiple markers within the same restriction

total_area = dolfinx.fem.form(one*dx_with_data((3, 7)))
outer_area = dolfinx.fem.form(one*dx_with_data(7))

# We can now assemble the forms as before.
# Since we will do this many times in this demo, we create a convenience function
# for assembling and accumulating a scalar value:

def assemble(form: ufl.Form|dolfinx.fem.Form)->dolfinx.default_scalar_type:
    compiled_form = dolfinx.fem.form(form)
    local_form = dolfinx.fem.assemble_scalar(compiled_form)
    return compiled_form.mesh.comm.allreduce(local_form, op=MPI.SUM)

# We also create a convenience function for computing the relative error
# between an approximate solution `a` and the exact solution `a_ex``:

def relative_error(a, a_ex):
    """Return the relative error in percent
    :param a: The approximate value
    :param a_ex: The exact value (cannot be 0)
    :return: Relative error in percent
    """
    return np.abs(a - a_ex)/a_ex*100

# We have that the area of the ellipsoid should be

A_ex_inner = np.pi*R_i*aspect_ratio*R_i

# We can now compare the computed areas to the exact areas

# ### Comparsion of inner area

ellipsoid_area = assemble(inner_area)
print(f"Number of elements: {linear_mesh.topology.index_map(linear_mesh.topology.dim).size_global}")
print(f"Inner area: {ellipsoid_area:.5e}, Exact: {A_ex_inner:.5e}\n",
      f"Relative error: {relative_error(ellipsoid_area, A_ex_inner):.2f}%")

# ### Comparsion of outer area

donut_area = assemble(outer_area)
print(f"Outer area: {donut_area:.5e}, Exact: {global_area - A_ex_inner:.5e}\n",
      f"Relative error: {relative_error(donut_area, A_ex-A_ex_inner):.2f}%")

# We observe quite large errors in the area computations.

# ### Comparison on refined mesh
# We create a refine mesh and compile the forms for the two domains

fine_linear_mesh, flct, ffct = generate_mesh(0.1, 1)
dx_fine = ufl.Measure("dx", domain=fine_linear_mesh, subdomain_data=flct)
inner_area = dolfinx.fem.form(one*dx_fine(3))
outer_area = dolfinx.fem.form(one*dx_fine(7))

# We assemble the scalar value as before

ellipsoid_area = assemble(inner_area)
donut_area = assemble(outer_area)
print(f"Number of elements: {fine_linear_mesh.topology.index_map(fine_linear_mesh.topology.dim).size_global}")
print(f"Area of the domain is {ellipsoid_area+donut_area:.5e}, expected {A_ex:.5e}\n", 
      f"Relative error: {relative_error(ellipsoid_area+donut_area, A_ex):.2f}%")
print(f"Inner area: {ellipsoid_area:.5e}, Exact: {A_ex_inner:.5e}\n",
      f"Relative error: {relative_error(ellipsoid_area, A_ex_inner):.2f}%")
print(f"Outer area: {donut_area:.5e}, Exact: {A_ex - A_ex_inner:.5e}\n",
      f"Relative error: {relative_error(donut_area, A_ex-A_ex_inner):.2f}%")

# However, how fine of a mesh do we need if we use third order elements?

# ### Comparsion on a curved mesh
# We use the coarsest mesh resolution from the above examples, and create a mesh with triangles with
# third order polynomials describing each facet.

curved_mesh, cct, cft = generate_mesh(0.5, 3)

# We repeat the process from above

dx_curved = ufl.Measure("dx", domain=curved_mesh, subdomain_data=cct)
inner_area = dolfinx.fem.form(one*dx_curved(3))
outer_area = dolfinx.fem.form(one*dx_curved(7))
ellipsoid_area = assemble(inner_area)
donut_area = assemble(outer_area)
print(f"Number of elements: {curved_mesh.topology.index_map(curved_mesh.topology.dim).size_global}")
print(f"Area of the domain is {ellipsoid_area+donut_area:.5e}, expected {A_ex:.5e}\n", 
      f"Relative error: {relative_error(ellipsoid_area+donut_area, A_ex):.2f}%")
print(f"Inner area: {ellipsoid_area:.5e}, Exact: {A_ex_inner:.5e}\n",
      f"Relative error: {relative_error(ellipsoid_area, A_ex_inner):.2f}%")
print(f"Outer area: {donut_area:.5e}, Exact: {A_ex - A_ex_inner:.5e}\n",
      f"Relative error: {relative_error(donut_area, A_ex-A_ex_inner):.2f}%")

# We observe that we get an extremely accurate estimate of the area.
# Should we therefore always use higher order meshes?

# #### Potential drawbacks of higher order grids
# In most problems, we are not just computing the surface area or volume of the mesh. 
# We usually have an unknown $u_h$ that we need to solve a PDE for.
#
# If we chose a **sub-parameteric approach**, where we use a lower order space for the unknown $u_h$ than
# for the mesh geometry, the solution will be poorly represented.
# There is therefore a balancing act between using a high resolution grid and higher order elements.
#
# It is usual to have higher order convergence rates when using higher order elements, so my rule of thumb is:
# If you use a higher order function-space for your unknown, and you are able to mesh your geometry with a
# higher order element, then do so.
# 
# The exception to this rule is when you are working with piecewise straight geometries, such as squares and boxes,
# as there is no benefit in using higher order elements for these geometries.
#
# ## Exercise
# 1. Do we really need a third order grid to represent the circular geometry accurately? Could we use a second order grid?
# 2. What happens to the integral of the constant function 1 over the domain if we use a second order or third order grid?


# # Integration over facets
# There are other integration measures that can be used in DOLFINx.
# 1. `"dx"` - Integration over all cells in your mesh
# 2. `"ds"` - Integration over all exterior facets in your mesh (All facets connected to only a single cell)
# 3. `"dS"` - Integration over all interior facets in your mesh (All facets connected to two cells)

# We can use the facet markers to integrate over the boundary of the domain

ds_linear = ufl.Measure("ds", domain=linear_mesh, subdomain_data=linear_facettags)

# For the example at hand, we only have one external boundary, which means that it is equivalent to write
# `ds_linear` or `ds_linear(15)`

# ## Example: Boundary integral with spatially varying functions
# In this example we will consider the boundary integral
#
# $$
# \int_{\partial\Omega} g n \cdot v ~\mathrm{d}s\qquad \forall v \in V,
# $$
# 
# where `g(x)` is a known function, `n` the outwards pointing facet normal and `v` a test function.

# We create a spatially varying function with `ufl.SpatialCoordinate`
# and use `ufl.FacetNormal` to symbolically describe the outward pointing normal

def linear_form(element, domain):
    x, y = ufl.SpatialCoordinate(domain)
    g = ufl.sin(x)*ufl.cos(y)
    n = ufl.FacetNormal(domain)
    V = dolfinx.fem.functionspace(domain, element)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=domain)
    return ufl.inner(g*n, v) * ds


# We start by considering a linear element

linear_el = ("Lagrange", 1, (2, ))

L1 = linear_form(linear_el, linear_mesh)

# Similarly, we create a form for the curved mesh

L2 = linear_form(linear_el, curved_mesh)

# ## Comparison of the integrals
# We can use UFL to analyze the integrals.
# One of the tools we can use is an estimator of the polynomial degree of the integrand

from ufl.algorithms import expand_derivatives, estimate_total_polynomial_degree

# ## Exercise
# - Do we expect the estimated polynomial degree to be the same for the two integrals?

# + tags=["hide-output"]
print(f"Linear mesh {estimate_total_polynomial_degree(expand_derivatives(L1))}")
print(f"Curved mesh {estimate_total_polynomial_degree(expand_derivatives(L2))}")
# -

# ```{admonition} Explanation
# :class: dropdown
# Since the mapping from the reference to the physical domain is no longer linear,
# we do not have a constant Jacobian. Similarly, the normal vector is no longer constant
# along the facet. This increases the polynomial degree of the integrand.
# ```

# ## Consequences
# Having a higher polynomial estimate means that we require more quadrature points
# to represent the integrand accurately.
# This can lead to a higher computational cost.

# We can use basix to investigate how many quadrature points there are in different default rules.

import basix
points, weights = basix.make_quadrature(linear_mesh.basix_cell(), 7, basix.QuadratureType.Default)
print(f"Number of quadrature points: {points.shape[0]}")

points, weights = basix.make_quadrature(linear_mesh.basix_cell(), 15, basix.QuadratureType.Default)
print(f"Number of quadrature points: {points.shape[0]}")

# This means that we will do three times the amount of computations on the curved mesh compared to the linear mesh.
# There is also the additional consequence that the Jacobian computation is moved into the quadrature loop of the assembly kernels.

# ## How to reduce the computational cost
# One way to reduce the computational cost is to fix the quadrature rule to a given order.
# We can do this through the metadata parameter in the `ufl.Measure` object.

dx_restricted = ufl.Measure("dx", domain=curved_mesh, metadata={"quadrature_degree": 7})

# This will override the estimated values by UFL.