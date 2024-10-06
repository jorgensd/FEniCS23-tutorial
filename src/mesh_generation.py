# # Mesh generation
#
# So far, we have covered how to generate code for the local element assembly.
# We can now look into how to solve specific problems.
# Let us start with generating the computational domain.
#
# In DOLFINx, the mesh creation requires 4 inputs:
#
# - **MPI communicator**: This is used to decide how the partitioning is performed.
#   It is usually `MPI.COMM_WORLD` or `MPI.COMM_SELF`.
# - **Nodes**: A set of coordinates in 1, 2 or 3D, that represents all the points in the mesh
# - **Connectivity**: A nested list, where each row corresponds to the node indices of a single cell
# - **Coordinate element**: A finite element used for pushing coordinates from the reference element
#   to the physical element and its inverse.
#
# As an example, let us consider a simple two element mesh, of two straight edged triangles
#

from mpi4py import MPI

import ipyparallel as ipp
import numpy as np
import pyvista

import basix.ufl
import dolfinx
import ufl

pyvista.start_xvfb(1.0)

# We start by creating the four nodes that the grid will consist of

nodes = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 2.0], [1, 3]], dtype=np.float64)

# Next, we define each cell by the index of the row each point has in the `nodes`-array

connectivity = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

# As we have seen in the [previous tutorial](./ufl_formulation) we use a finite element to
# describe the mapping from the reference triangles to the physical triangles described above.

c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(nodes.shape[1],)))

# Finally we create a mesh object by calling `dolfinx.mesh.create_mesh` with the four
# aforementioned inputs

domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, connectivity, nodes, c_el)

# The mesh can be visualized with Paraview, or Pyvista.
# Following is a short code for visualizing the mesh with Pyvista


def plot_mesh(mesh: dolfinx.mesh.Mesh):
    """
    Given a DOLFINx mesh, create a `pyvista.UnstructuredGrid`,
    and plot it and the mesh nodes
    """
    plotter = pyvista.Plotter()
    V_linear = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    linear_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V_linear))
    if mesh.geometry.cmap.degree > 1:
        ugrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
        plotter.add_mesh(ugrid, style="points", color="b", point_size=10)
        ugrid = ugrid.tessellate()
        plotter.add_mesh(ugrid, show_edges=False)
    plotter.add_mesh(linear_grid,show_edges=True)

    plotter.show_axes()
    plotter.view_xy()
    plotter.show()


# The mesh we created above is visualized as

plot_mesh(domain)


# ## Higher order meshes
# As we use a finite element to describe the mapping from the reference element to the physical element,
# we can use higher order elements to describe the reference element, giving us the possibility to create curved meshes.
# In the following example,w e will create a single cell mesh (triangle) using a
# [second order Lagrange element](https://defelement.com/elements/examples/triangle-lagrange-equispaced-2.html).
#
# This finite element has a total of 6 degrees of freedom, and we will use the standard dual basis:
#
# $$
# \begin{align}
# l_0:& v \mapsto v(0, 0) \\
# l_1:& v \mapsto v(1, 0) \\
# l_2:& v \mapsto v(0, 1) \\
# l_3:& v \mapsto v(0.5, 0.5) \\
# l_4:& v \mapsto v(0, 0.5) \\
# l_6:& v \mapsto v(0.5, 0)
# \end{align}
#
# We will create a set of six nodes, where we follow the ordering of the dual basis functions

nodes = np.array(
    [[1.0, 0.0], [2.0, 0.0], [3.0, 2.0], [2.5, 1], [1.5, 1.5], [1.5, -0.2]],
    dtype=np.float64,
)
connectivity = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)

c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 2, shape=(nodes.shape[1],)))
domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, connectivity, nodes, c_el)

# ### Questions/Exercises
# - Where are the point evaluations $l_3, l_4, l_5$ located in the reference triangle?
# - What entities do we associated each functional with?
# - Can you make a visualization script for the push forward from the reference element to the a physical element,
#   as done in  {ref}`straight_edge_triangle`?

# Press the following dropdown to reveal the solution
# + tags=["hide-input"]
def compute_physical_point(points,  X, degree: int = 2):
    """
    Map coordinates `X` in reference element to triangle defined by `p0`, `p1` and `p2`
    """
    el = basix.ufl.element("Lagrange", "triangle", degree)
    basis_values = el.tabulate(0, X)
    assert points.shape[0] == basis_values.shape[2], "Nodes not matching finite element basis functions"
    return (basis_values[0] @ points)
#-
# + tags=["hide-input"]
theta = 2 * np.pi
reference_points = basix.create_lattice(basix.CellType.triangle, 9,
                                        basix.LatticeType.equispaced, exterior=True)
x = compute_physical_point(nodes, reference_points, degree=2)
phi = np.linspace(0, theta, reference_points.shape[0])
rgb_cycle = (np.stack((np.cos(phi),
                       np.cos(phi-theta/4),
                       np.cos(phi+theta/4)
                      )).T
             + 1)*0.5 # Create a unique colors for each node
import matplotlib.pyplot as plt
fig, (ax_ref, ax) = plt.subplots(2, 1)
# Plot reference points
reference_vertices = basix.cell.geometry(basix.CellType.triangle)
ref_triangle= plt.Polygon(reference_vertices, color="blue", alpha=0.2)
ax_ref.add_patch(ref_triangle)
ax_ref.scatter(reference_points[:,0], reference_points[:,1], c=rgb_cycle)
# Plot physical points
triangle = plt.Polygon(nodes[[0,1,2]], color="blue", alpha=0.2)
# Plot all nodes
ax.scatter(nodes[:,0], nodes[:, 1], color="black", marker="s")
ax.add_patch(triangle)
ax.scatter(x[:,0], x[:,1], c=rgb_cycle)
# -

plot_mesh(domain)

# ## Interfacing with other software
# As DOLFINx works on `numpy`-arrays it is easy to convert any mesh format that be converted into this structure.
# DOLFINx has a `gmsh`-interface, using the {term}`GMSH` Python-API to read `GMSH`-models or `.msh` files.
# We will start with creating a simple circular geometry using GMSH.

import gmsh

# The first thing we do when using GMSH is to initialize it explicitly

gmsh.initialize()

# A single GMSH instance can create multiple separate geometries, named models.
# We start by creating a model

gmsh.model.add("circle")

# Next, we use the Open Cascade {term}`OCC` backend to create a circle

center_x = 0.5
center_y = 1
radius_x = 0.8
radius_y = 0.3
circle = gmsh.model.occ.addDisk(center_x, center_y, 0, radius_x, radius_y)

# What GMSH returns is an integer, which internally represents a geometry object of topological dimension two.
# We will later use this integer to mark the surface as something we would like to output.
#
# ```{note}
# The circle object has no concept of mesh resolution.
# This is controlled at a later stage with the mesh algorithm, where a finer resolution will
# more closely approximate the circle. 2D meshing algorithms uses the parameterization of the
# circle to create the mesh.
# ```

# To make it possible for other functions in GMSH than those in the OCC module to modify
# the `circle`, we need to synchronize the model.

gmsh.model.occ.synchronize()

# In GMSH, you can combine multiple geometries into a single mesh (by unions, intersections or differences).
# In order to ensure that only the parts of the mesh that you would like to use is generated (and no overlapping cells),
# you need to create Physical Groups for all volumes, surfaces and curves you would like to have in your mesh.

# We will start by adding the circle to a physical group.

# + tags=["remove-output"]
gmsh.model.addPhysicalGroup(2, [circle], tag=3)
# -

# In the above code we have specified the topological dimension of the entity (a surface).
# We have specified the list of GMSH objects that should be tagged, and we have given these objects
# the marker with value 3.

# Next, we would like to mark the external boundary, such that we can apply boundary conditions to it.
# We start by getting all the boundary entities of the circle

boundary_curves = gmsh.model.getBoundary([(2, circle)], recursive=False, oriented=False)

# As output of `getBoundary` we get a list of tuples, where each tuple contains the topological
# dimension of the entity, and the integer that represents the entity.
# In this case, we only have a single boundary curve, and we will mark it with the tag 4.

# + tags=["remove-output"]
assert len(boundary_curves) == 1, "Expected a single boundary curve"
gmsh.model.addPhysicalGroup(boundary_curves[0][0], [boundary_curves[0][1]], tag=4)
# -

# Next, we can generate the mesh

gmsh.model.mesh.generate(2)

# In GMSH we can choose to mesh first, second or third order meshes. We choose a 3rd order mesh for the circle.

gmsh.model.mesh.setOrder(3)

# The mesh is generated, but not yet saved to disk. We could save it to disk using `gmsh.write` command, or import it
# directly into DOLFINx.
# We choose the latter, and use `dolfinx.io.gmshio.model_to_mesh` to convert the GMSH model to a DOLFINx mesh.
# For this function we need to send in a few objects:
# 1. The `gmsh.model` instance
# 2. The `MPI` communicator we want to distribute the mesh over
# 3. The `MPI` rank that holds the mesh
#    ```{note}
#    GMSH does not have a concept of an MPI distributed mesh, and will generate a mesh on each process if used as above.
#    Usually, one would call `gmsh.initialize()` on all processors, then add an `if-else` clause that only generates
#    the mesh on a single processor (usually rank 0).
#    ```
# 4. The geometrical dimension of the mesh when used in DOLFINx
#   ```{note}
#    GMSH always generates three-dimensional points, which means that a 1D or 2D mesh could be interpreted as a manifold in
#    a higher dimensional space. The `gdim` argument is used to specify the dimension of the mesh when used in DOLFINx.
#   ```

circular_mesh, cell_marker, facet_marker = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

# We can now finalize GMSH (as we will not use it further in this section), and inspect the cell_markers and facet_markers

gmsh.finalize()

print(f"Topological dimension of input cells of the mesh {cell_marker.dim}, should match {circular_mesh.topology.dim}")
print(f"Value of marked cells {cell_marker.values}")
print(f"List of facets (index local to process) which are on the boundary {facet_marker.indices}",
      f"with corresponding values {facet_marker.values}")

plot_mesh(circular_mesh)

# # Extra topic: Reading in meshes in parallel
# For more information about mesh-partitioning and mesh input, see: https://jsdokken.com/dolfinx_docs/meshes.html
# ## Cell ownership
#
# For the remainder of this section we will consider a 3x3 unit square mesh:

domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)

# The mesh consists of cells, edges and vertices.
# A mesh is created by supplying the information regarding the connectivity between the cells and the mesh nodes.
# As a higher order mesh has more nodes than vertices, we can get the connectivity between the cells and mesh vertices
# through the mesh-topology.

# We start by creating a simple function for inspecting these outputs in serial and parallel


def inspect_mesh(shared_facet: bool = False):
    from mpi4py import MPI

    import dolfinx

    ghost_mode = dolfinx.mesh.GhostMode.shared_facet if shared_facet else dolfinx.mesh.GhostMode.none
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, ghost_mode=ghost_mode)
    topology = domain.topology
    tdim = topology.dim
    print(f"Number of cells in process: {topology.index_map(tdim).size_local}", flush=True)
    print(f"Number of shared cells: {topology.index_map(tdim).num_ghosts}", flush=True)
    print(f"Global range {topology.index_map(tdim).local_range}")
    cell_to_vertices = topology.connectivity(tdim, 0)
    print(cell_to_vertices)


# We start by inspecting the outputs in serial

inspect_mesh()

# We observe that we have 18 cells in the mesh, that are connected to the 18 vertices.
# All indices start from 0, and can be mapped to its global owner by calling `domain.topology.index_map(tdim).local_to_global([idx])`

# ## Parallel execution
# We use IPython-Parallel for running DOLFINx on multiple processes inside our script.

cluster = ipp.Cluster(engines="mpi", n=2)
rc = cluster.start_and_connect_sync()

# This has started to processes that can execute code with a MPI communicator.
# We run the mesh code on two processes, instructing DOLFINx not to share any cells between the processes

query = rc[:].apply_async(inspect_mesh, False)
query.wait()
query.display_outputs()

# Next, we instruct DOLFINx to share cells between two processes if a facet is shared between the two processes.

new_query = rc[:].apply_async(inspect_mesh, True)
new_query.wait()
new_query.display_outputs()
# + tags=["hide-output"]
cluster.stop_cluster_sync()
