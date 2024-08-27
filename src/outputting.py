# # Outputting

# In the previous sections we have only considered continuous and discontinuous Lagrange elements.
# However, there is a large variety of elements that is supported in DOLFINx.

# + tags=["hide-input"]
from IPython.display import IFrame

IFrame(
    "https://defelement.com/lists/implementations/basix.ufl.html",
    width=900,
    height=1000,
)
# -

# Many of these elements have functionals that are not point evaluations, and
# they span a large amount of different polynomials.

# VTK, used in visualization in Pyvista and Paraview does not support most of these elements.
# However, they [support arbitrary order Lagrange elements](http://www.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/), which means that if we can interpolate
# or project the solution we would like to output into a P-th order space that is compatible,
# we can use Paraview/VTK to visualize the solution.

# The engines `VTXWriter` and `VTKFile` in DOLFINx to write output compatible with Paraview.
