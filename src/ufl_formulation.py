# # The Unified Form Language
# We have previously seen how to define a finite element, and evaluate its basis functions in points on the
# reference element. However, in this course we aim to solve problems from solid mechanics.
# Thus, we need more than the basis functions to efficiently solve the problems at hand.
# In this section, we will introduce the unified form language (UFL), which is a domain-specific language for
# defining variational formulations for partial differential equations.
# The goal of this section is to be able to solve the following problem:
#
# Given to function $f$ and $g$, compute the approximation of $\frac{f}{g}$ in a specified finite
# element space.
# Mathematically, we can write
#
# Find $u\in V(\Omega)$ such that
# \begin{align}
# u &= \frac{f(x,y,z)}{g(x,y,z)} \qquad \text{in } \Omega\subset \mathbb{R}^3.
# \end{align}
#
# We start by focusing on the computational domain $\Omega$,

# ## The computational domain
# For a 3D problem, we can represent the computational domain by either tetrahedral or hexahedral elements.
# For a 2D problem, we can use either triangular or quadrilateral elements.
#
# One of the fundamental ideas in the Finite Element method is to map the integrals from the physical domain
# to the reference domain element by element, and then insert the local contribution in the global system.
# The required operations for this mapping is to be able to compute the Jacobian, its inverse and determinant
# for any of the cells above, when represented by points $p_0, \dots, p_M$, $M$ being the number of
# nodes describing the cell.


# In this example, we will use straight edged triangular elements, which we note that we can represent by
# a first order Lagrange element.

import basix.ufl
import ufl

cell = "triangle"
c_el = basix.ufl.element("Lagrange", cell, 1, shape=(2,))

# Note that if we wanted to represent a 2D manifold represented in 3D, we could change the `shape`-parameter to `(3, )`.
# We call this element the coordinate element, as it represents the transformation of coordinates back and forth from the
# reference element.

# In the unified form language we make explicit representations of the computational domain, using `ufl.Mesh`.
domain = ufl.Mesh(c_el)

# We note that this is a purely symbolic representation, it doesn't matter if we are solving something on a
# unit square, a circle or a 2D slice through a brain.

# ## The function space
# As opposed some commercial software, we do not rely on iso-parameteric elements in FEniCS.
# We can use any supported finite element space to describe our unknown `u`.

el = basix.ufl.element("Discontinuous Lagrange", cell, 2)
V = ufl.FunctionSpace(domain, el)

# For the coefficients `f` and `g`, we choose

F = ufl.FunctionSpace(domain, basix.ufl.element("Discontinuous Lagrange", cell, 0))
f = ufl.Coefficient(F)
G = ufl.FunctionSpace(domain, basix.ufl.element("Lagrange", cell, 1))
g = ufl.Coefficient(G)


# ## The variational form
# We obtain them in the standard way, by multiplying by a test function
# and integrating over the domain, we do this by using `dx`, which means that we integrate over all cells of the mesh.

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(u, v) * ufl.dx

# Note that the bi-linear form `a` is defined with trial and test functions.

L = (f / g) * v * ufl.dx
forms = [a, L]

# So far, so good?
# As opposed to most demos/tutorials on FEniCSx, note that we have not imported `dolfinx`
# or made a reference to the actual computational domain we want to solve the problem on or what `f` or `g` is,
# except for the choice of function spaces.

# ## Further analysis of the variational form
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
# A_{j, i} &= \int_{K_{ref}} \phi_i(M_K(\bar x))\phi_j(M_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x\\
# b_j &= \int_{K_{ref}} \frac{\Big(\sum_{k=0}^{\mathcal{M}}f_k\psi_i(M_K(\bar x))\Big)}
# {\Big(\sum_{l=0}^{\mathcal{T}}g_k\varphi_i(M_K(\bar x))\Big)}\phi_j(M_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x
# \end{align}
# Next, one can choose an appropriate quadrature rule with points and weights, include the
# correct mapping/restrictions of degrees of freedom for each cell.
# All of this becomes quite tedious and error prone work, and has to be repeated for every variational form!

pulled_back_L = ufl.algorithms.compute_form_data(
    L,
    do_apply_function_pullbacks=True,
    do_apply_integral_scaling=True,
    do_apply_geometry_lowering=True,
    preserve_geometry_types=(ufl.classes.Jacobian,),
)
print(pulled_back_L.integral_data[0])


# # Functionals and derivatives
# In the unified form language, we can also compute the derivatives of the variational form with respect to coefficients.
# This is essential for efficient implementations of optimization and inverse problems.
# We start by defining a functional
#
# $$J_h(u_h) = \int_\Omega \sigma(u_h): \epsilon(u_h)~\mathrm{d}x,$$
#
# where $\sigma$ is the stress tensor, $\epsilon$ is the symmetric strain tensor and $u_h$ a displacement field.
#
# We start by defining these quantities in UFL:

# The function space for displacement

el = basix.ufl.element("Lagrange", cell, 1, shape=(2,))
Vh = ufl.FunctionSpace(domain, el)
uh = ufl.Coefficient(Vh)

# Lame's elasticity parameters

mu = ufl.Constant(domain)
lmbda = ufl.Constant(domain)

# The stress and strain tensor


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    return 2 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))


# The functional

Jh = ufl.inner(sigma(uh), epsilon(uh)) * ufl.dx

# Usually, a functional in itself isn't very interesting, but we can compute the derivative of the functional with respect to the displacement field.

dJhdu = ufl.derivative(Jh, uh)

# Given the equations of linear elasticity, we coould also compute the adjoint of the derivative of the residual with respect to the displacement field.,
# This is needed for solving the adjoint equations in optimization problems.

vh = ufl.TestFunction(Vh)
F = ufl.inner(sigma(uh), epsilon(vh)) * ufl.dx
dFdu_adj = ufl.adjoint(ufl.derivative(F, uh))

forms = [a, L, Jh, dJhdu, dFdu_adj]
