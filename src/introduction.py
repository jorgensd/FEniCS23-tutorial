# + [markdown] editable=true slideshow={"slide_type": "slide"}
# # Choosing and using a finite element
# As seen in DOLFINx in general, we use the unified form language (UFL) to
# define variational forms.
# The power of this domain specific language is that it resembles mathematical
# syntax.
#
# We will start with a standard problem, namely a projection:
# \begin{align}
# u &= \frac{f(x,y,z)}{g(x,y,z)} \qquad \text{in } \Omega\subset \mathbb{R}^3.
# \end{align}
# where $\Omega$ is our computational domain, $f$ and $g$ are two known functions.
#
# + [markdown] slideshow={"slide_type": "slide"}
# ## Finite elements
# + [markdown] slideshow={"slide_type": "skip"}
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
# 2) A set of input points to evaluate the basis functions at as a numpy array of shape
# # `(num_points, reference_cell_dimension)`.
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
