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
# <figure>
# <img src="https://defelement.com/img/element-Lagrange-variant-equispaced-triangle-1-0-large.png" width="150" height="150" />
# <img src="https://defelement.com/img/element-Lagrange-variant-equispaced-triangle-1-1-large.png" width="150" height="150" />
# <img src="https://defelement.com/img/element-Lagrange-variant-equispaced-triangle-1-2-large.png" width="150" height="150" />
# <figcaption> First order Lagrange basis functions <figcaption\>
# <figure\>
#
# To symbolically represent this element, we use `ufl.FiniteElement`

# + slideshow={"slide_type": ""}
import numpy as np
import basix.ufl
import ufl

element = ufl.FiniteElement("Lagrange", ufl.triangle, 1)

# -
# We note that we send in the finite element family, what cells we will use to represent the domain,
# and the degree of the function space.
# As this element is just a symbolic representation, it does not contain the information required to tabulate (evaluate)
# basis functions at an abitrary point. UFL is written in this way, so that it can be used as a symbolic representation
# for a large range of finite element software, not just FENiCS.
#
# In FENiCSx, we use [Basix](https://github.com/FEniCS/basix/) to tabulate finite elements.
# Basix can convert any `ufl`-element into a basix finite element, which we in turn can evaluate.

basix_element = basix.ufl.convert_ufl_element(element)

# Lets next tabulate the basis functions of our element at a given point `p=(0, 0.5)` in the reference element.
# the `basix.finite_element.FiniteElement.tabulate` function takes in two arguments, how many derivatives we want to compute,
# and at what points in the reference element we want to compute them.

points = np.array([[0., 0.5], [1, 0]], dtype=np.float64)
print(basix_element.tabulate(0, points))

# We can also compute the derivatives at any points in the reference element

print(basix_element.tabulate(1, points))

# Observe that the output we get from this command also includes the 0th order derivatives.
# Thus we note that the output has the shape `(num_spatial_derivatives+1, num_points, num_basis_functions)`
#
# Not every function we want to represent is scalar valued. For instance, in fluid flow problems, the Taylor-Hood finite element
# pair is often used to represent the fluid velocity and pressure, where each component of the fluid velocity is in a Lagrange space.
# We represent this by using a `ufl.VectorElement`, which you can give the number of components as an input argument.

vector_element = ufl.VectorElement("Lagrange", ufl.triangle, 2, dim=2)

# We can also use basix for this

el = basix.ufl.element("Lagrange", "triangle", 2)
v_el = basix.ufl.blocked_element(el, shape=(2,))

# Both these definitions of elements are compatible with DOLFINx. The reason for having both of them is that basix offers more tweaking of the finite
# elements that the UFL definition does. For more information regarding this see:
# [Variants of Lagrange elements](https://docs.fenicsproject.org/dolfinx/v0.7.2/python/demos/demo_lagrange_variants.html)

# To create the Taylor-Hood finite element pair, we use the `ufl.MixedElement`/`basix.ufl.mixed_element` syntax

m_el = ufl.MixedElement([vector_element, element])
m_el_basix = basix.ufl.mixed_element([v_el, basix_element])

# There is a wide range of finite elements that are supported by ufl and basix.
# See for instance: [Supported elements in basix/ufl](https://defelement.com/lists/implementations/basix.ufl.html).
