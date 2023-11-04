# # An introduction to the unified form language
# As seen in DOLFINx in general, we use the unified form language (UFL) to
# define variational forms.
# The power of this domain specific language is that it resembles mathematical
# syntax.
#
# We will start with a standard problem, namely a projection:
# \begin{align}
# u &= \frac{f(x,y,z)}{g(x,y,z)} \qquad \text{in } \Omega(x,y,z).
# \end{align}
# where $\Omega$ is our computational domain, $f$ and $g$ are two known functions
#
# ## Finite elements
# To solve this problem, we have to choose an approporiate finite element space to represent the function $k$ and $u$.
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

# +
import numpy as np
import basix.ufl_wrapper
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

basix_element = basix.ufl_wrapper.convert_ufl_element(element)

# Lets next tabulate the basis functions of our element at a given point `p=(0, 0.5)` in the reference element.
# the `basix.finite_element.FiniteElement.tabulate` function takes in two arguments, how many derivatives we want to compute,
# and at what points in the reference element we want to compute them.

point = np.array([[0., 0.5], [1, 0]], dtype=np.float64)
print(basix_element.tabulate(0, point))

# We can also compute the derivatives at any point in the reference element

print(basix_element.tabulate(1, point))

# Observe that the output we get from this command also includes the 0th order derivatives.
# Thus we note that the output has the shape (num_spatial_derivatives+1, num_points, num_basis_functions)
#
# Not every function we want to represent is scalar valued. For instance, in fluid flow problems, the Taylor-Hood finite element
# pair is often used to represent the fluid velocity and pressure, where each component of the fluid velocity is in a Lagrange space.
# We represent this by using a `ufl.VectorElement`, which you can give the number of components as an input argument.

vector_element = ufl.VectorElement("Lagrange", ufl.triangle, 2, dim=2)

# We can also use basix for this

el = basix.ufl_wrapper.create_element("Lagrange", "triangle", 2)
v_el = basix.ufl_wrapper.VectorElement(el, size=2)

# Both these definitions of elements are compatible with DOLFINx. The reason for having both of them is that basix offers more tweaking of the finite
# elements that the UFL definition does. For more information regarding this see:
# [Variants of Lagrange elements](https://docs.fenicsproject.org/dolfinx/v0.6.0/python/demos/demo_lagrange_variants.html)

# To create the Taylor-Hood finite element pair, we use the `ufl.MixedElement`/`basix.ufl_wrapper.MixedElement` syntax

m_el = ufl.MixedElement([vector_element, element])
m_el_basix = basix.ufl_wrapper.MixedElement([v_el, basix_element])

# There is a wide range of finite elements that are supported by ufl and basix.
# See for instance: [Supported elements in basix/ufl](https://defelement.com/lists/implementations/basix.ufl.html).

#
# ## The computational domain
# Now that we have covered how to define finite elements, we can move on
# to the computational domain $\Omega$.
# We can use either tetrahedral and hexahedral elements to subdivide the continuous
# domain into a discrete domain. We define a coordinate element, which is used to map
# functions defined on the reference element to the physical domain.
# In this example, we will use straight edged hexahedral elements, and therefore the coordinate element
# can be defined as

c_el = ufl.VectorElement("Lagrange", ufl.hexahedron, 1)

# Next, we create an abstract definition of the computational domain. For now, we don't know if we are solving
# the Poisson equation on a cube, a sphere or on the wing of an airplane.

domain = ufl.Mesh(c_el)

# ## The function space
# As oposed some commerical software, we do not rely on iso-parameteric elements in FEniCS.
# We can use any supported finite element space to describe our unknown `u`.

el = ufl.FiniteElement("DQ", domain.ufl_cell(), 2)
V = ufl.FunctionSpace(domain, el)

# For the coefficients `f` and `g`, we choose

F = ufl.FunctionSpace(domain, ufl.FiniteElement("DQ", domain.ufl_cell(), 0))
f = ufl.Coefficient(F)
G = ufl.FunctionSpace(domain, ufl.FiniteElement(
    "Lagrange", domain.ufl_cell(), 1))
g = ufl.Coefficient(F)


# ## The variational form
# We obtain them in the standard way, by multiplying by a test function
# and integrating over the .domain

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(u, v) * ufl.dx

# Note that the bi-linear form `a` is defined with trial and test functions.

L = (f / g) * v * ufl.dx

# So far, so good?
# As opposed to most demos/tutorials on FEniCSx, note that we have not imported `dolfinx` or made a reference to the actual
# computational domain we want to solve the problem on or what `f` or `g` is,
# except for the choice of function spaces


# # Code generation
# All the code above is Python code symbolically describing the variational form of the projection.

# To do so, we would define each of the functions as the linear
# combinations of the basis functions
# $u=\sum_{i=0}^{\mathcal{N}}u_i\phi_i(x)\qquad
# v=\sum_{i=0}^{\mathcal{N}}v_i\phi_i(x)\qquad
# f=\sum_{k=0}^{\mathcal{M}}f_k\psi_k(x)\qquad
# g=\sum_{l=0}^{\mathcal{T}}g_l\varphi_l(x)$
#
# We next use the map $M_K:K_{ref}\mapsto K$
# \begin{align}
# \int_\Omega u v~\mathrm{d}x&= \sum_{K\in\mathcal{K}}\int_K u(x) v(x)~\mathrm{d}x\\
# &= \sum_{K\in\mathcal{K}}\int_{M_K(K_{ref})} u(x)v(x)~\mathrm{d}x\\
# &= \sum_{K\in\mathcal{K}}\int_{K_{ref}}u(M_K(\bar x))v(M_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x
# \end{align}
# where $K$ is each element in the physical space, $J_K$ the Jacobian of the mapping.
# Next, we can insert the expansion of $u$ into the formulation and identify the matrix system $Au=b$, where
# \begin{align}
# A_{j, i} &= \int_{K_{ref}} \phi_i(M_K(\bar x))\phi_j(M_K(\bar x))~\mathrm{d}\bar x\\
# b_j &= \int_{K_{ref}} \frac{\Big(\sum_{k=0}^{\mathcal{M}}f_k\psi_i(M_K(\bar x))\Big)}
# {\Big(\sum_{l=0}^{\mathcal{T}}g_k\varphi_i(M_K(\bar x))\Big)}\phi_j(M_K(\bar x))~\mathrm{d}\bar x
# \end{align}
# Next, one can choose an appropriate quadrature rule with points and weights
