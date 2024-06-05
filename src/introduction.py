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
# where $\Omega$ is our computational domain, $f$ and $g$ are two known functions
#
# + [markdown] slideshow={"slide_type": "slide"}
# ## Finite elements
# + [markdown] slideshow={"slide_type": "skip"}
# To solve this problem, we have to choose an appropriate finite element space to represent the function $k$ and $u$.
# There is a large variety of finite elements, for instance the [Lagrange elements](https://defelement.com/elements/lagrange.html).
# The basis function for a first order Lagrange element is shown below
#
#<center> 
# <img src="https://defelement.com/img/element-Lagrange-variant-equispaced-triangle-1-0-large.png" width="150" height="150" />
# <img src="https://defelement.com/img/element-Lagrange-variant-equispaced-triangle-1-1-large.png" width="150" height="150" />
# <img src="https://defelement.com/img/element-Lagrange-variant-equispaced-triangle-1-2-large.png" width="150" height="150" /><br>
# First order Lagrange basis functions<br><br>
# </center>

#
# To symbolically represent this element, we use `basix.ufl.element`, that creates a UFL-compatible element using Basix.

# + slideshow={"slide_type": ""}
import numpy as np
import basix.ufl

element = basix.ufl.element("Lagrange", "triangle", 1)

# -
# We note that we send in the finite element family, what cells we will use to represent the domain,
# and the degree of the function space.
# As this element is just a symbolic representation, it does not contain the information required to tabulate (evaluate)
# basis functions at an arbitrary point. UFL is written in this way, so that it can be used as a symbolic representation
# for a large range of finite element software, not just FENiCS.
#

# Lets next tabulate the basis functions of our element at a given point `p=(0, 0.5)` in the reference element.
# the `basix.finite_element.FiniteElement.tabulate` function takes in two arguments, how many derivatives we want to compute,
# and at what points in the reference element we want to compute them.

points = np.array([[0., 0.5], [1, 0]], dtype=np.float64)
print(element.tabulate(0, points))

# We can also compute the derivatives at any points in the reference element

print(element.tabulate(1, points))

# Observe that the output we get from this command also includes the 0th order derivatives.
# Thus we note that the output has the shape `(num_spatial_derivatives+1, num_points, num_basis_functions)`
#
# Not every function we want to represent is scalar valued. For instance, in fluid flow problems, the Taylor-Hood finite element
# pair is often used to represent the fluid velocity and pressure, where each component of the fluid velocity is in a Lagrange space.
# We represent this by adding a `shape` argument to the `basix.ufl.element` constructor.

vector_element = basix.ufl.element("Lagrange", "triangle", 2, shape=(2, ))

# Basix allows for a large variety of extra options to tweak your finite elements, see for instance
# [Variants of Lagrange elements](https://docs.fenicsproject.org/dolfinx/v0.8.0/python/demos/demo_lagrange_variants.html)
# for how to choose the node spacing in a Lagrange element.

# To create the Taylor-Hood finite element pair, we use the `basix.ufl.mixed_element`

m_el = basix.ufl.mixed_element([vector_element, element])

# There is a wide range of finite elements that are supported by ufl and basix.
# See for instance: [Supported elements in basix/ufl](https://defelement.com/lists/implementations/basix.ufl.html).
