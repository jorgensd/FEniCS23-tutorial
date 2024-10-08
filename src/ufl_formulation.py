# # Creating a variational formulation in the Unified Form Language ({term}`UFL`)
#
# We have previously seen how to define a finite element, and evaluate its basis functions in points on the
# reference element.
# However, in this course we aim to solve problems from solid mechanics.
# Thus, we need more than the basis functions to efficiently solve the problems at hand.
#
# In this section, we will introduce the Unified Form Language {term}`UFL`, which is a domain-specific language for
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
#
# ## The computational domain
# The general idea of the finite element method is to sub-divide $\Omega$ into
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
# (straight_edge_triangle)=
# ## Example: Straight edged triangle
#
# As we saw in [the section on finite elements](./introduction), we can use basix to get a
# sample of points within the reference element.

import basix.ufl
import numpy as np
reference_points = basix.create_lattice(basix.CellType.triangle, 13,
                                        basix.LatticeType.gll, exterior=False,
                                        method=basix.LatticeSimplexMethod.warp)

# Next, we realize that we can use the first order Lagrange space, to represent the mapping from the
# reference element to any physical element:
# Given three points, $p_0=(x_0, y_0)^T, p_1=(x_1,y_1)^T, p_2=(x_2,y_2)^T$, we can represent any point $x$
# as the linear combination of the three basis functions on the reference element $X$.
#
# $$x = F_j(X)= \sum_{i=0}^3 p_i \phi_i(X).$$
#

def compute_physical_point(p0, p1, p2, X):
    """
    Map coordinates `X` in reference element to triangle defined by `p0`, `p1` and `p2`
    """
    el = basix.ufl.element("Lagrange", "triangle", 1)
    basis_values = el.tabulate(0, X)
    return (basis_values[0] @ np.vstack([p0, p1, p2]))

# We can now experiment with this code

p0 = np.array([2.0, 1.4])
p1 = np.array([1.0, 1.2])
p2 = np.array([1.3, 1.0])
x = compute_physical_point(p0, p1, p2, reference_points)

# We use matplotlib to visualize the reference points and the physical points

import matplotlib.pyplot as plt
theta = 2 * np.pi
phi = np.linspace(0, theta, reference_points.shape[0])
rgb_cycle = (np.stack((np.cos(phi),
                       np.cos(phi-theta/4),
                       np.cos(phi+theta/4)
                      )).T
             + 1)*0.5 # Create a unique colors for each node

fig, (ax_ref, ax) = plt.subplots(2, 1)
# Plot reference points
reference_vertices = basix.cell.geometry(basix.CellType.triangle)
ref_triangle= plt.Polygon(reference_vertices, color="blue", alpha=0.2)
ax_ref.add_patch(ref_triangle)
ax_ref.scatter(reference_points[:,0], reference_points[:,1], c=rgb_cycle)
# Plot physical points
vertices = np.vstack([p0, p1, p2])
triangle = plt.Polygon(vertices, color="blue", alpha=0.2)
ax.add_patch(triangle)
ax.scatter(x[:,0], x[:,1], c=rgb_cycle)

# ## Exercises:
#
# - Can we use a similar kind of mapping on a quadrilateral/tetrahedral/hexahedral element?
# - What happens if we change the order of the basis functions?
# - How can we compute the Jacobian of the mapping?

# ## Using UFL to define a symbolic representation of a domain
# As seen above, we can use the Lagrange element basis functions to represent the mapping from the
# reference element to the physical element (and its inverse).

# In the unified form language we make a symbolic representation of the computational domain, using `ufl.Mesh`.

import basix.ufl
import ufl

cell = "triangle"
c_el = basix.ufl.element("Lagrange", cell, 1, shape=(2,))

# Note that if we wanted to represent a 2D manifold represented in 3D, we could change the `shape`-parameter to `(3, )`.
# We call this element the coordinate element, as it represents the transformation of coordinates between the reference
# element and the physical element.

domain = ufl.Mesh(c_el)

# We note that this is a purely symbolic representation, it doesn't matter if we are solving something on a
# unit square, on a 2D representation of a bridge or a brain. The only thing that matters is what kind of elements we
# use to represent the domain.

# ## The function space
#
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
# As opposed some commercial software, we do not rely on iso-parametric elements in FEniCS.
# We can use any supported finite element space to describe our unknown `u`.
#
# For more advanced maps, see for instance:
# [DefElement - vector valued basis functions](https://defelement.com/ciarlet.html#Vector-valued+basis+functions).
#

el = basix.ufl.element("Discontinuous Lagrange", cell, 2)
V = ufl.FunctionSpace(domain, el)

# For the coefficients `f` and `g`, we choose

F = ufl.FunctionSpace(domain, basix.ufl.element("Discontinuous Lagrange", cell, 0))
f = ufl.Coefficient(F)
G = ufl.FunctionSpace(domain, basix.ufl.element("Lagrange", cell, 1))
g = ufl.Coefficient(G)

# (variational_form)=
# ## The variational form of a projection
#
#
# ```{admonition} The finite element method and its links to minimization
# Recall that the variational form is a way of writing the {term}`PDE` on a form that we can use to solve the problem with
# {term}`FEM`.
# We start by multiplying the equation with a test-function from a suitable space (in this case the same space as the solution)
# and integrate over the domain.
#
# $$ \int_\Omega u\cdot v~\mathrm{d}x = \int_\Omega \frac{f}{g}\cdot v~\mathrm{d}x.$$
#
# This is also equivalent to minimizing the following functional
# 
# $$ \min_{u\in V} J(u) = \frac{1}{2}\int_\Omega \left(u-\frac{f}{g}\right)\cdot\left(u-\frac{f}{g}\right)~\mathrm{d}x.$$
#
# We find the minimum by computing $\frac{\mathrm{d}{J}}{\mathrm{d}u}[\delta u]=0$.
#
# $$ \frac{\mathrm{d}{J}}{\mathrm{d}u}[\delta u] = \int_\Omega \left(u - \frac{f}{g}\right)\cdot \delta u~\mathrm{d}x=0.$$
#
#```
#
#
# We obtain them in the standard way, by multiplying by a test function
# and integrating over the domain, we do this by using `dx`, which means that we integrate over all cells of the mesh.

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(u, v) * ufl.dx

# Note that the bi-linear form `a` is defined with trial and test functions.

L = ufl.inner(f / g, v) * ufl.dx
forms = [a, L]

# So far, so good?
# As opposed to most demos/tutorials on FEniCSx, note that we have not imported `dolfinx`
# or made a reference to the actual computational domain we want to solve the problem on or what `f` or `g` is,
# except for the choice of function spaces.

# ## Further analysis of the variational form
# We next use the map $F_K:K_{ref}\mapsto K$ to map the integrals over each cell in the domain back to the reference cell.
# \begin{align}
# \int_\Omega u v~\mathrm{d}x&= \sum_{K\in\mathcal{K}}\int_K u(x) v(x)~\mathrm{d}x\\
# &= \sum_{K\in\mathcal{K}}\int_{F_K(K_{ref})} u(x)v(x)~\mathrm{d}x\\
# &= \sum_{K\in\mathcal{K}}\int_{K_{ref}}u(F_K(\bar x))v(F_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x\\
# \int_\Omega \frac{f}{g}v~\mathrm{d}x
# &=\sum_{K\in\mathcal{K}}\int_{K_{ref}}\frac{f(F_K(\bar x))}{g(F_K(\bar x))} v(F_K(\bar x)) \vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x
# \end{align}
# where $K$ is each element in the physical space, $J_K$ the Jacobian of the mapping.

#
# Next, we can insert the expansion of $u, v, f, g$ into the formulation:
# $u=\sum_{i=0}^{\mathcal{N}}u_i\phi_i(x)\qquad
# v=\sum_{i=0}^{\mathcal{N}}v_i\phi_i(x)\qquad
# f=\sum_{k=0}^{\mathcal{M}}f_k\psi_k(x)\qquad
# g=\sum_{l=0}^{\mathcal{T}}g_l\varphi_l(x)$
#

#
#  and identify the matrix system $Au=b$, where
# \begin{align}
# A_{j, i} &= \int_{K_{ref}} \phi_i(F_K(\bar x))\phi_j(F_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x\\
# b_j &= \int_{K_{ref}} \frac{\Big(\sum_{k=0}^{\mathcal{M}}f_k\psi_i(F_K(\bar x))\Big)}
# {\Big(\sum_{l=0}^{\mathcal{T}}g_k\varphi_i(F_K(\bar x))\Big)}\phi_j(F_K(\bar x))\vert \mathrm{det} J_K(\bar x)\vert~\mathrm{d}\bar x
# \end{align}
#
# ```{warning}
# Next, one can choose an appropriate quadrature rule with points and weights, include the
# correct mapping/restrictions of degrees of freedom for each cell.
# All of this becomes quite tedious and error prone work, and has to be repeated for every variational form!
# ```

# Since this is time consuming work, we will use {term}`UFL` to compute these operations for us.
# Below is an example of how we can use UFL on the variational form above to apply the pull back to the reference element.

pulled_back_L = ufl.algorithms.compute_form_data(
    L,
    do_apply_function_pullbacks=True,
    do_apply_integral_scaling=True,
    do_apply_geometry_lowering=True,
    preserve_geometry_types=(ufl.classes.Jacobian,),
)
print(pulled_back_L.integral_data[0])

# (functionals)=
# # Functionals and derivatives
# As mentioned above, many finite element problems can be rephrased as an optimization problem.
#
# For instance, we can write the equations of linear elasicity as an optimization problem:
#
# $$\min_{u_h\in V}J_h(u_h) = \int_\Omega C\epsilon(u_h): \epsilon(u_h)~\mathrm{d}x - \int_\Omega f\cdot v~\mathrm{d}x,$$
#
# where $C$ is the stiffness tensor given as $C_{ijkl} = \lambda \delta_{ij}\delta_{kl} + \mu(\delta_{ik}\delta_{jl}+\delta_{il}\delta{kj})$,
# $\epsilon$ is the symmetric strain tensor and $u_h$ a displacement field.
#
# We start by defining these quantities in UFL:

# The function space for displacement

el = basix.ufl.element("Lagrange", cell, 1, shape=(2,))
Vh = ufl.FunctionSpace(domain, el)
uh = ufl.Coefficient(Vh)
f = ufl.Coefficient(Vh)

# Lame's elasticity parameters

mu = ufl.Constant(domain)
lmbda = ufl.Constant(domain)

def epsilon(u):
    return ufl.sym(ufl.grad(u))


# We define the stiffness tensor using [Einstein summation notation](https://mathworld.wolfram.com/EinsteinSummation.html).
# We start by defining the identity tensor which we will use as a [Kronecker Delta](https://mathworld.wolfram.com/KroneckerDelta.html)
# function.
# Next we define four indices that we will use to account for the four dimensions of the stiffness tensor.

Id = ufl.Identity(domain.geometric_dimension())
indices = ufl.indices(4)

# Secondly we define the product of two delta functions $\delta_{ij}\delta_{kl}$
# which results in a fourth order tensor.

def delta_product(i, j, k, l):
    return ufl.as_tensor(Id[i, j] * Id[k, l], indices)

# Finally we define the Stiffness tensor
i,j,k,l = indices
C = lmbda * delta_product(i,j,k,l) + mu*(delta_product(i,k,j,l) + delta_product(i,l,k,j))

# and the functional

Jh = 0.5*(C[i,j,k,l] * epsilon(uh)[k,l]) * epsilon(uh)[i,j] * ufl.dx - ufl.inner(f, uh) * ufl.dx

# This syntax is remarkably similar to how it is written on [paper](https://en.wikipedia.org/wiki/Elasticity_tensor).

# ## Alternative formulation
# Instead of writing out all the indices with Einstein notation, one could write the same equation as

def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

Jh = 0.5 * ufl.inner(sigma(uh), epsilon(uh)) * ufl.dx - ufl.inner(f, uh) * ufl.dx

# ## Differentiating the energy functional
# We can differentiate the energy functional with respect to the displacement field $u_h$.

F = ufl.derivative(Jh, uh)

# Since we want to find the minimum of the functional, we set the derivative to zero.
# To solve this problem, we can for instance use a Newton method, where we solve a sequence of equations:
#
# $$ u_{k+1} = u_k - J_F(u_k)^{-1}F(u_k),$$
#
# where $J_F$ is the Jacobian matrix of $F$.
# We can rewrite this as:
#
# $$
# \begin{align}
# u_{k+1} &= u_k - \delta u_k\\
# J_F(u_k)\delta u_k &= F(u_k)
# \end{align}
# $$
#
# Which boils down to solving a linear system of equations for $\delta u_k$.
#
# We can compute the Jacobian using UFL:

J_F = ufl.derivative(F, uh)

# And with this and $F$ we can solve the minimization problem.
# See for instance:
# [Custom Newton Solver in DOLFINx](https://jsdokken.com/dolfinx-tutorial/chapter4/newton-solver.html)
# for more details about how you could implement this method by hand.
#
# # Extra material: Optimization problems with PDE constraints
#
# Another class of optimization problems is the so-called PDE-constrained optimization problems.
# For these problems, one usually have a problem on the form
#
# $$\min_{c}J_h(u_h, c)$$
#
# such that
#
# $$F(u_h,c)=0.$$
#
# We can use the adjoint method to compute the sensitivity of the functional with respect to the solution of the PDE.
# This is done by introducing the Lagrangian
#
# $$\min_{c}\mathcal{L}(u_h, c) = J_h(u_h,c) + (\lambda, F(u_h,c)).$$
#
# We now seek the minimum of the Lagrangian, i.e.
#
# $$\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}c}[\delta c] = 0,$$
#
# which we can write as
#
# $$
# \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}c}[\delta c]
# = \frac{\partial J}{\partial c}
# + \frac{\partial J}{\partial u}\frac{\mathrm{d}u}{\mathrm{d}c}[\delta c]
# + \left(\lambda, \frac{\partial F}{\partial u} \frac{\mathrm{d}u}{\mathrm{d}c}[\delta c]\right)
# + \left(\lambda, \frac{\partial F}{\partial c}[\delta c]\right).
# $$
#
# Since $\lambda$ is arbitrary, we choose $\lambda$ such that
#
# $$
# \frac{\partial J}{\partial u}\delta u
# = -\left(\lambda, \frac{\partial F}{\partial u} \delta u\right)
# $$
#
# for any $\delta u$ (including $\frac{\mathrm{d}u}{\mathrm{d}c}[\delta c]$).
#
# This would mean that
#
# $$
# \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}c}[\delta c]
# = \frac{\partial J}{\partial c}  + \left(\lambda, \frac{\partial F}{\partial c}[\delta c]\right).
# $$
#
# To find such a $\lambda$, we can solve the adjoint problem
#
# $$
# \left( \left(\frac{\partial F}{\partial u}\right)^* \lambda^*, \delta u\right) =
# -\left(\frac{\partial J}{\partial u}\right)^*\delta u.
# $$
#
# With UFL, we do not need to derive these derivatives by hand, and can use symbolic
# differentiation to get the left and right hand side of the adjoint problem.

dFdu_adj = ufl.adjoint(ufl.derivative(F, uh))
dJdu_adj = ufl.derivative(Jh, uh)
dJdf = ufl.derivative(Jh, f)
dFdf = ufl.derivative(F, f)
