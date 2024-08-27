# # Recap of the finite element method
# The finite element method is a way of representing a function $u$ in a function space $V$, given $u$
# satisfies a certain partial differential equation.
#
# In this tutorial, we will keep to function spaces
# $V\subset H^{1}(\Omega)$, $H^1(\Omega):=\{u\in L^2(\Omega) \text{ and } \nabla u\in L^2(\Omega)\}$.
#
# Next, we choose a finite subset of functions in $H^1(\Omega)$, and we call this the finite element space.
# This means that any function in $V$ can be represented as a linear combination of these basis functions
#
# $$u\in V \Leftrightarrow u(x) = \sum_{i=1}^{N} u_i \phi_i(x),$$
#
# where $N$ are the number of basis functions, $\phi_i$ is the $i$-th basis function and $u_i$ are the coefficients.
#
# How the basis functions of interest are chosen is often dependent on the physical problem at hand.
#
# The general idea of the finite element method is to sub-divide the computational domain into
# smaller (polygonal) elements $K_j$ such that
# 1) The triangulation covers $\Omega$: $\cup_{j=1}^{M}K_j=\bar{\Omega}$
# 2) No overlapping polyons: $\mathrm{int} K_i \cap \mathrm{int} K_j=\emptyset$ for $i\neq j$.
# 3) No vertex lines in the interior of a facet or edge of another element
#
# We will call our polygonal domain $\mathcal{K}={K_j}_{j=1}^{M}$.
# Next, we define a reference element $K_{ref}$, which is a simple polygon that we can map to any element $K_j$,
# using the mapping $F_j:K_{ref}\mapsto K_j$.
#
# We define the Jacobian of this mapping as $\mathbf{J_j}$.
#
# ## Selecting a discrete function space
# Once we have subdivided $\Omega$ into elements $K$, we can define a discrete function space $V_h$:
#
# $$V_h=\{v \in H^1(\mathcal{K})\}.$$
#
# Certain finite elements need to conserve certain properties (0-valued normals components of facets, etc).
# We define this map as: $(\mathcal{F}_j(\phi))(x)$.
#
# For the finite elements we will consider in this tutorial, we will use the map
#
# $$(\mathcal{F}_j(\phi))(x) = \hat\phi(F_j^{-1}(x)).$$
#
# where $\hat\phi$ is the basis function on the reference element.
# In other words, to evaluate the basis function in a point in the physical space, we pull the point back
# to the reference element evaluate our local basis function at this point
#
# Thus, we can write out the evaluation of a finite element function as
#
# $$u(x)=\sum_{i=1}^N u_i\phi_i(F_j^{-1}(x)).$$
#
# For more advanced maps, see for instance:
# [DefElement - vector valued basis functions](https://defelement.com/ciarlet.html#Vector-valued+basis+functions).
#
# # Creating a finite element in FEniCSx
#
# To solve this problem, we have to choose an appropriate finite element space to represent the function $k$ and $u$.
# There is a large variety of finite elements, for instance the
# [Lagrange elements](https://defelement.com/elements/lagrange.html).
# The basis function for a first order Lagrange element is shown below
#
# <center>
# <img src="https://defelement.com/img/element-Lagrange-variant-equispaced-triangle-1-0-large.png"
# width="150" height="150" />
# <img src="https://defelement.com/img/element-Lagrange-variant-equispaced-triangle-1-1-large.png"
# width="150" height="150" />
# <img src="https://defelement.com/img/element-Lagrange-variant-equispaced-triangle-1-2-large.png"
# width="150" height="150" /><br>
# First order Lagrange basis functions<br><br>
# </center>

#
# To symbolically represent this element, we use `basix.ufl.element`,
# that creates a UFL-compatible element using [Basix](https://github.com/FEniCS/basix/).

# + slideshow={"slide_type": ""}
import numpy as np

import basix.ufl

element = basix.ufl.element("Lagrange", "triangle", 1)

# -
# The three arguments we provide to Basix are the name of the element family, the reference
# cell we would like to use and the degree of the polynomial space.
#

# Next, we will evaluate the basis functions at a certain set of `points` in the reference triangle.
# We call this tabulation and we use the `tabulate` method of the element object.
# The two input arguments are:
# 1) The number of spatial derivatives of the basis functions we want to compute.
# 2) A set of input points to evaluate the basis functions at as
# a numpy array of shape `(num_points, reference_cell_dimension)`.
#
# In this case, we want to compute the basis functions themselves, so we set the first argument to 0.

points = np.array([[0.0, 0.5], [1, 0]], dtype=np.float64)
print(element.tabulate(0, points))

# We can also compute the derivatives at any points in the reference element

print(element.tabulate(1, points))

# Observe that the output we get from this command also includes the 0th order derivatives.
# Thus we note that the output has the shape `(num_spatial_derivatives+1, num_points, num_basis_functions)`
#
# Not every function we want to represent is scalar valued.
# For instance, in fluid flow problems, the Taylor-Hood finite element pair is often used
# to represent the fluid velocity and pressure.
# For the velocity, each component (x, y, z) is represented with its own degrees of freedom in a Lagrange space..
# We represent this by adding a `shape` argument to the `basix.ufl.element` constructor.

vector_element = basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))

# Basix allows for a large variety of extra options to tweak your finite elements, see for instance
# [Variants of Lagrange elements](https://docs.fenicsproject.org/dolfinx/v0.8.0/python/demos/demo_lagrange_variants.html)
# for how to choose the node spacing in a Lagrange element.

# To create the Taylor-Hood finite element pair, we use the `basix.ufl.mixed_element`

m_el = basix.ufl.mixed_element([vector_element, element])

# There is a wide range of finite elements that are supported by ufl and basix.
# See for instance: [Supported elements in basix/ufl](https://defelement.com/lists/implementations/basix.ufl.html).

# ## Lower precision tabulation
#
# In some cases, one might want to use a lower accuracy for tabulation of basis functions to speed up computations.
# This can be changed in basix by adding `dtype=np.float32` to the element constructor.

low_precision_element = basix.ufl.element("Lagrange", "triangle", 1, dtype=np.float32)
points_low_precision = points.astype(np.float32)
basis_values = low_precision_element.tabulate(0, points_low_precision)
print(f"{basis_values=}\n   {basis_values.dtype=}")

# We observe that elements that are close to zero is now an order of magnitude larger than its `np.float64` counterpart.
